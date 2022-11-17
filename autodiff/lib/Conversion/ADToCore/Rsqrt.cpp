#include "Rule/Helper.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

Value calRsqrtGrad(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
                   PatternRewriter& rewriter) {
  auto rsqrt = createOp<math::RsqrtOp>(rewriter, resultTypes, args);
  return getGradient(rsqrt, ones(rewriter, rsqrt), args[0]);
}

class RsqrtToCore : public OpRewritePattern<ad::RsqrtOp> {
  using OpRewritePattern<ad::RsqrtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::RsqrtOp rsqrt,
                                PatternRewriter& rewriter) const override {
    auto resultType = rsqrt.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      return elementwiseMatchAndRewriteHelper(rsqrt, rewriter, calRsqrtGrad);
    }

    auto value = calRsqrtGrad(rsqrt, rsqrt.getX(), resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(rsqrt, value);
    return success();
  }
};

}  // namespace mlir::autodiff
