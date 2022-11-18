#include "Dialect/Grad/IR/Grad.hpp"
#include "Rewriter.hpp"

namespace mlir::autodiff {

Value clampHelper(ValueRange args, PatternRewriter& rewriter,
                  Value leUpperBound, Value geLowerBound) {
  auto x = args[0];
  auto dout = args[1];

  // begin if
  auto isLe = createOp<scf::IfOp>(rewriter, x.getType(), leUpperBound, true);

  // begin [x > max]
  auto gtUpperBuilder = isLe.getElseBodyBuilder();
  createOp<scf::YieldOp>(gtUpperBuilder, zeros(gtUpperBuilder, x));
  // end [x > max]

  // begin [x <= max]
  auto leUpperBuilder = isLe.getThenBodyBuilder();
  auto isGe =
      createOp<scf::IfOp>(leUpperBuilder, x.getType(), geLowerBound, true);

  // begin [x < min]
  auto ltLowerBuilder = isGe.getElseBodyBuilder();
  createOp<scf::YieldOp>(ltLowerBuilder, zeros(ltLowerBuilder, x));
  // end [x < min]

  // begin [min <= x <= max]
  auto geLowerBuilder = isGe.getThenBodyBuilder();
  createOp<scf::YieldOp>(geLowerBuilder, ones(geLowerBuilder, x));
  // end [min <= x <= max]

  createOp<scf::YieldOp>(leUpperBuilder, isGe->getResult(0));
  // end if

  auto value = isLe->getResult(0);
  return product(rewriter, dout, value);
}

Value clampIntGrad(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
                   PatternRewriter& rewriter) {
  auto min = op->getAttr("min_int").cast<IntegerAttr>();
  auto max = op->getAttr("max_int").cast<IntegerAttr>();
  auto minValue = createOp<arith::ConstantOp>(rewriter, min);
  auto maxValue = createOp<arith::ConstantOp>(rewriter, max);

  auto x = args[0];

  auto leUpperBound =
      createOp<arith::CmpIOp>(rewriter, arith::CmpIPredicate::sle, x, maxValue);
  auto geLowerBound =
      createOp<arith::CmpIOp>(rewriter, arith::CmpIPredicate::sge, x, minValue);

  return clampHelper(args, rewriter, leUpperBound, geLowerBound);
}

Value clampFloatGrad(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
                     PatternRewriter& rewriter) {
  auto min = op->getAttr("min_fp").cast<FloatAttr>();
  auto max = op->getAttr("max_fp").cast<FloatAttr>();
  auto minValue = createOp<arith::ConstantOp>(rewriter, min);
  auto maxValue = createOp<arith::ConstantOp>(rewriter, max);

  auto x = args[0];

  auto leUpperBound =
      createOp<arith::CmpFOp>(rewriter, arith::CmpFPredicate::OLE, x, maxValue);
  auto geLowerBound =
      createOp<arith::CmpFOp>(rewriter, arith::CmpFPredicate::OGE, x, minValue);

  return clampHelper(args, rewriter, leUpperBound, geLowerBound);
}

class ClampToCore : public OpRewritePattern<grad::ClampOp> {
  using OpRewritePattern<grad::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::ClampOp clamp,
                                PatternRewriter& rewriter) const override {
    auto resultType = clamp.getType().dyn_cast<TensorType>();

    if (!resultType) return failure();

    auto elemType = resultType.getElementType();

    if (isa<IntegerType>(elemType))
      return elementwiseMatchAndRewriteHelper(clamp, rewriter, clampIntGrad);

    else
      return elementwiseMatchAndRewriteHelper(clamp, rewriter, clampFloatGrad);
  }
};

}  // namespace mlir::autodiff
