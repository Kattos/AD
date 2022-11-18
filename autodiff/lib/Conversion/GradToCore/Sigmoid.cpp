#include "Rewriter.hpp"

namespace mlir::autodiff {

Value sigmoidGrad(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
                  PatternRewriter& rewriter) {
  // \derivative{sigmoid(x)}{x} = \frac{e^{-x}}{(1 + e^{-x})^2}
  auto x = args[0];
  auto dout = args[1];

  auto minusX = createOp<arith::NegFOp>(rewriter, x);
  auto exp = createOp<math::ExpOp>(rewriter, minusX);
  auto sqrtDominant = sum(rewriter, ones(rewriter, exp), exp);
  auto dominant = product(rewriter, sqrtDominant, sqrtDominant);

  auto grad = createOp<arith::DivFOp>(rewriter, exp, dominant);
  return product(rewriter, dout, grad);
}

class SigmoidToCore : public OpRewritePattern<grad::SigmoidOp> {
  using OpRewritePattern<grad::SigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::SigmoidOp sigmoid,
                                PatternRewriter& rewriter) const override {
    // note that `math.exp` support float only
    auto resultType = sigmoid.getType();

    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      auto elemType = shapedType.getElementType();

      if (isa<IntegerType>(elemType)) return failure();

      return elementwiseMatchAndRewriteHelper(sigmoid, rewriter, sigmoidGrad);
    }

    if (isa<IntegerType>(resultType)) return failure();

    if (auto value =
            sigmoidGrad(sigmoid, sigmoid.getOperands(), resultType, rewriter)) {
      rewriter.replaceOp(sigmoid, value);
      return success();
    }

    return failure();
  }
};

}  // namespace mlir::autodiff
