#include "Rewriter.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::autodiff {

// cal binary grad in unary way
class AddToCore : public OpRewritePattern<grad::AddOp> {
  using OpRewritePattern<grad::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::AddOp add,
                                PatternRewriter& rewriter) const override {
    // cal grad of binary op in an unary way
    auto dout = add.getDout();
    auto lhs = add.getLhs();
    auto lhsType = lhs.getType();
    auto rhs = add.getRhs();
    auto rhsType = rhs.getType();

    auto dlhs = createOp<grad::AddLhsOp>(rewriter, lhsType, lhs, rhs, dout);
    auto drhs = createOp<grad::AddRhsOp>(rewriter, rhsType, rhs, rhs, dout);

    if (!dlhs || !rhs) {
      return failure();
    }

    rewriter.replaceOp(add, {dlhs, drhs});
    return success();
  }
};

class AddLhsToCore : public OpRewritePattern<grad::AddLhsOp> {
  using OpRewritePattern<grad::AddLhsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::AddLhsOp lhs,
                                PatternRewriter& rewriter) const override {
    auto resultType = lhs.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();
      auto dout = lhs.getDout();
      auto doutForLhs = reduce(rewriter, dout, lhs.getLhs());
      lhs->setOperand(2, doutForLhs);

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(lhs, rewriter,
                                                lhsCalFn<arith::AddIOp>);

      else
        return elementwiseMatchAndRewriteHelper(lhs, rewriter,
                                                lhsCalFn<arith::AddFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value =
          lhsCalFn<arith::AddIOp>(lhs, lhs.getOperands(), resultType, rewriter);

    else
      value =
          rhsCalFn<arith::AddFOp>(lhs, lhs.getOperands(), resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(lhs, value);
    return success();
  }
};

class AddRhsToCore : public OpRewritePattern<grad::AddRhsOp> {
  using OpRewritePattern<grad::AddRhsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::AddRhsOp rhs,
                                PatternRewriter& rewriter) const override {
    auto resultType = rhs.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();
      auto dout = rhs.getDout();
      auto doutForRhs = reduce(rewriter, dout, rhs.getRhs());
      rhs->setOperand(2, doutForRhs);

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(rhs, rewriter,
                                                rhsCalFn<arith::AddIOp>);

      else
        return elementwiseMatchAndRewriteHelper(rhs, rewriter,
                                                rhsCalFn<arith::AddFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value =
          rhsCalFn<arith::AddIOp>(rhs, rhs.getOperands(), resultType, rewriter);

    else
      value =
          rhsCalFn<arith::AddFOp>(rhs, rhs.getOperands(), resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(rhs, value);
    return success();
  }
};

void populateAddToCore(RewritePatternSet& patterns) {
  patterns.add<AddToCore, AddLhsToCore, AddRhsToCore>(patterns.getContext());
}

}  // namespace mlir::autodiff
