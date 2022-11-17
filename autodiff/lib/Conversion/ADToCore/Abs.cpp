#include "Conversion/Conversion.hpp"
#include "Rule/Helper.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

template <typename AbsTy>
Value calAbsGrad(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
                 PatternRewriter& rewriter) {
  auto abs = createOp<AbsTy>(rewriter, resultTypes, args);
  return getGradient(abs, ones(rewriter, abs), args[0]);
}

class AbsToCore : public OpRewritePattern<ad::AbsOp> {
  using OpRewritePattern<ad::AbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::AbsOp abs,
                                PatternRewriter& rewriter) const override {
    auto resultType = abs.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(abs, rewriter,
                                                calAbsGrad<math::AbsIOp>);

      else
        return elementwiseMatchAndRewriteHelper(abs, rewriter,
                                                calAbsGrad<math::AbsFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value = calAbsGrad<math::AbsIOp>(abs, abs.getX(), resultType, rewriter);

    else
      value = calAbsGrad<math::AbsFOp>(abs, abs.getX(), resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(abs, value);
    return success();
  }
};

}  // namespace mlir::autodiff
