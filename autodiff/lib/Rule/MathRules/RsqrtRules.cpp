#include "MathRules.hpp"

namespace mlir::autodiff {

template <>
Value MathRsqrtRule::getInputDerivative(OpBuilder& builder,
                                        math::RsqrtOp rsqrt) {
  auto x = rsqrt.getOperand();
  auto type = x.getType();

  auto coefficientAttr = builder.getFloatAttr(type, -0.5);
  auto coefficient = createOp<arith::ConstantOp>(builder, coefficientAttr);

  auto exponentAttr = builder.getFloatAttr(type, -1.5);
  auto exponent = createOp<arith::ConstantOp>(builder, exponentAttr);

  auto pow = createOp<math::PowFOp>(builder, x, exponent);
  return product(builder, coefficient, pow);
}

}  // namespace mlir::autodiff
