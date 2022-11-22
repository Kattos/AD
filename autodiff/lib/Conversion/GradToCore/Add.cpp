#include "Rewriter.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::autodiff {

class AddLhsToCore : public OpRewritePattern<grad::AddLhsOp> {
  using OpRewritePattern<grad::AddLhsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::AddLhsOp lhsOp,
                                PatternRewriter& rewriter) const override {
    auto resultType = lhsOp.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();
      auto lhs = lhsOp.getLhs();

      auto dout = lhsOp.getDout();
      auto doutForLhs = reduce(rewriter, dout, lhs);
      lhsOp->setOperand(2, doutForLhs);

      auto rhs = lhsOp.getRhs();
      auto rhsForLhs = reduce(rewriter, rhs, lhs);
      lhsOp->setOperand(1, rhsForLhs);

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(lhsOp, rewriter,
                                                lhsCalFn<arith::AddIOp>);

      else
        return elementwiseMatchAndRewriteHelper(lhsOp, rewriter,
                                                lhsCalFn<arith::AddFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value = lhsCalFn<arith::AddIOp>(lhsOp, lhsOp.getOperands(), resultType,
                                      rewriter);

    else
      value = lhsCalFn<arith::AddFOp>(lhsOp, lhsOp.getOperands(), resultType,
                                      rewriter);

    if (!value) return failure();

    rewriter.replaceOp(lhsOp, value);
    return success();
  }
};

class AddRhsToCore : public OpRewritePattern<grad::AddRhsOp> {
  using OpRewritePattern<grad::AddRhsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::AddRhsOp rhsOp,
                                PatternRewriter& rewriter) const override {
    auto resultType = rhsOp.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();
      auto rhs = rhsOp.getRhs();

      auto dout = rhsOp.getDout();
      auto doutForRhs = reduce(rewriter, dout, rhs);
      rhsOp->setOperand(2, doutForRhs);

      auto lhs = rhsOp.getLhs();
      auto lhsForRhs = reduce(rewriter, lhs, rhs);
      rhsOp->setOperand(0, lhsForRhs);

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(rhsOp, rewriter,
                                                rhsCalFn<arith::AddIOp>);

      else
        return elementwiseMatchAndRewriteHelper(rhsOp, rewriter,
                                                rhsCalFn<arith::AddFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value = rhsCalFn<arith::AddIOp>(rhsOp, rhsOp.getOperands(), resultType,
                                      rewriter);

    else
      value = rhsCalFn<arith::AddFOp>(rhsOp, rhsOp.getOperands(), resultType,
                                      rewriter);

    if (!value) return failure();

    rewriter.replaceOp(rhsOp, value);
    return success();
  }
};

void populateAddToCore(RewritePatternSet& patterns) {
  patterns.add<AddToCore, AddLhsToCore, AddRhsToCore>(patterns.getContext());
}

}  // namespace mlir::autodiff
