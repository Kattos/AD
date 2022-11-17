#include "MathRules.hpp"

namespace mlir::autodiff {

Value ln(OpBuilder& builder, Type type, double x) {
  auto attr = builder.getFloatAttr(type, x);
  auto xValue = createOp<arith::ConstantOp>(builder, attr);
  return createOp<math::LogOp>(builder, xValue);
}

Value reciprocal(OpBuilder& builder, Value value) {
  auto one = ones(builder, value);
  return createOp<arith::DivFOp>(builder, one, value);
}

Value gradLog(OpBuilder& builder, Operation* op, Value lnx) {
  auto coefficient = reciprocal(builder, lnx);
  auto xReciprocal = reciprocal(builder, op->getOperand(0));
  return product(builder, coefficient, xReciprocal);
}

template <>
Value MathLogRule::getInputDerivative(OpBuilder& builder, math::LogOp log) {
  // 1 = ln(e)
  return gradLog(builder, log, ones(builder, log));
}

template <>
Value MathLog10Rule::getInputDerivative(OpBuilder& builder,
                                        math::Log10Op log10) {
  auto ln10 = ln(builder, log10.getType(), 10.0);
  return gradLog(builder, log10, ln10);
}

template <>
Value MathLog2Rule::getInputDerivative(OpBuilder& builder, math::Log2Op log2) {
  auto ln2 = ln(builder, log2.getType(), 2.0);
  return gradLog(builder, log2, ln2);
}

template <>
Value MathLog1pRule::getInputDerivative(OpBuilder& builder,
                                        math::Log1pOp log1p) {
  auto x = log1p.getOperand();
  auto one = ones(builder, log1p);
  auto x1 = sum(builder, x, one);
  return reciprocal(builder, x1);
}

}  // namespace mlir::autodiff
