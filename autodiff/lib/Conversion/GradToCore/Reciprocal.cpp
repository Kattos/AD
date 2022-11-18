#include "Rewriter.hpp"

namespace mlir::autodiff {

Value reciprocalIntGrad(Operation* op, ValueRange args,
                        ArrayRef<Type> resultTypes, PatternRewriter& rewriter) {
  auto x = args[0];
  auto dout = args[1];
  auto square = product(rewriter, x, x);

  auto minusOneAttr = rewriter.getI64IntegerAttr(-1);
  auto minusOne = createOp<arith::ConstantOp>(rewriter, minusOneAttr);

  auto negate = product(rewriter, square, minusOne);
  auto grad =
      createOp<arith::DivSIOp>(rewriter, ones(rewriter, negate), negate);
  return product(rewriter, dout, grad);
}

Value reciprocalFloatGrad(Operation* op, ValueRange args,
                          ArrayRef<Type> resultTypes,
                          PatternRewriter& rewriter) {
  auto x = args[0];
  auto dout = args[1];
  auto square = product(rewriter, x, x);

  auto minusOneAttr = rewriter.getF32FloatAttr(-1.0);
  auto minusOne = createOp<arith::ConstantOp>(rewriter, minusOneAttr);

  auto negate = product(rewriter, square, minusOne);
  auto grad = createOp<arith::DivFOp>(rewriter, ones(rewriter, negate), negate);
  return product(rewriter, dout, grad);
}

class ReciprocalToCore : public OpRewritePattern<grad::ReciprocalOp> {
  using OpRewritePattern<grad::ReciprocalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::ReciprocalOp reciprocal,
                                PatternRewriter& rewriter) const override {
    auto resultType = reciprocal.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(reciprocal, rewriter,
                                                reciprocalIntGrad);

      else
        return elementwiseMatchAndRewriteHelper(reciprocal, rewriter,
                                                reciprocalFloatGrad);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value = reciprocalIntGrad(reciprocal, reciprocal.getOperands(),
                                resultType, rewriter);

    else
      value = reciprocalFloatGrad(reciprocal, reciprocal.getOperands(),
                                  resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(reciprocal, value);
    return success();
  }
};

}  // namespace mlir::autodiff
