#include "MathRules.hpp"

namespace mlir::autodiff {

// TODO: solve precision problem
template <>
Value MathTanhRule::getInputDerivative(OpBuilder& builder, math::TanhOp tanh) {
  // \derivative{tanh(x)}{x} = \frac{4}{(e^x + e^{-x})^2}
  auto x = tanh.getOperand();
  auto minusX = createOp<arith::NegFOp>(builder, x);
  auto eX = createOp<math::ExpOp>(builder, x);
  auto eMinusX = createOp<math::ExpOp>(builder, minusX);

  auto tmp = sum(builder, eX, eMinusX);
  auto dominator = product(builder, tmp, tmp);

  auto four = builder.getF32FloatAttr(4.0);
  auto numerator = createOp<arith::ConstantOp>(builder, four);

  return createOp<arith::DivFOp>(builder, numerator, dominator);
}

}  // namespace mlir::autodiff
