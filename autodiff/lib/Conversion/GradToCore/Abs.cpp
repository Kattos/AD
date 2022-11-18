#include "Rewriter.hpp"

namespace mlir::autodiff {

class AbsToCore : public OpRewritePattern<grad::AbsOp> {
  using OpRewritePattern<grad::AbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::AbsOp abs,
                                PatternRewriter& rewriter) const override {
    auto resultType = abs.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(abs, rewriter,
                                                unaryCalFn<math::AbsIOp>);

      else
        return elementwiseMatchAndRewriteHelper(abs, rewriter,
                                                unaryCalFn<math::AbsFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value = unaryCalFn<math::AbsIOp>(abs, abs.getOperands(), resultType,
                                       rewriter);

    else
      value = unaryCalFn<math::AbsFOp>(abs, abs.getOperands(), resultType,
                                       rewriter);

    if (!value) return failure();

    rewriter.replaceOp(abs, value);
    return success();
  }
};

}  // namespace mlir::autodiff
