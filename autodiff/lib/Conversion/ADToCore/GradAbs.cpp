#include "Conversion/Conversion.hpp"
#include "Rule/Helper.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

template <typename AbsTy>
Value gradAbs(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
              PatternRewriter& rewriter) {
  auto x = args[0];
  auto grad = args[1];
  auto abs = createOp<AbsTy>(rewriter, resultTypes, x);

  auto outputGrad = getGradient(abs, ones(rewriter, grad), args[0]);
  return product(rewriter, grad, outputGrad);
}

class GradAbsToCore : public OpRewritePattern<ad::GradAbsOp> {
  using OpRewritePattern<ad::GradAbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::GradAbsOp abs,
                                PatternRewriter& rewriter) const override {
    auto resultType = abs.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();

      if (isa<IntegerType>(elemType))
        return elementwiseMatchAndRewriteHelper(abs, rewriter,
                                                gradAbs<math::AbsIOp>);

      else
        return elementwiseMatchAndRewriteHelper(abs, rewriter,
                                                gradAbs<math::AbsFOp>);
    }

    Value value;

    if (isa<IntegerType>(resultType))
      value =
          gradAbs<math::AbsIOp>(abs, abs.getOperands(), resultType, rewriter);

    else
      value =
          gradAbs<math::AbsFOp>(abs, abs.getOperands(), resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(abs, value);
    return success();
  }
};

}  // namespace mlir::autodiff
