#include "TosaRules.hpp"

namespace mlir::autodiff {

ValueRange getTosaGradients(Operation* op, Value grad) {
  if (isa<tosa::ExpOp>(op))
    return getGradients<TosaExpRule>(op, grad);

  else if (isa<tosa::LogOp>(op))
    return getGradients<TosaLogRule>(op, grad);

  else if (isa<tosa::AddOp>(op))
    return getGradients<TosaAddRule>(op, grad);

  else if (isa<tosa::SubOp>(op))
    return getGradients<TosaSubRule>(op, grad);

  else if (isa<tosa::MulOp>(op))
    return getGradients<TosaMulRule>(op, grad);

  else
    assert(false && "Unsupported `tosa` operation detected");
}

Value getTosaGradient(Operation* op, Value grad, Value input) {
  if (isa<tosa::ExpOp>(op))
    return getGradient<TosaExpRule>(op, grad, input);

  else if (isa<tosa::LogOp>(op))
    return getGradient<TosaLogRule>(op, grad, input);

  else if (isa<tosa::AddOp>(op))
    return getGradient<TosaAddRule>(op, grad, input);

  else if (isa<tosa::SubOp>(op))
    return getGradient<TosaSubRule>(op, grad, input);

  else if (isa<tosa::MulOp>(op))
    return getGradient<TosaMulRule>(op, grad, input);

  else
    assert(false && "Unsupported `tosa` operation detected");
}

template <>
Value TosaExpRule::getInputDerivative(OpBuilder& builder, tosa::ExpOp exp) {
  auto x = exp.getInput1();
  return product(builder, exp, x);
}

template <>
Value TosaLogRule::getInputDerivative(OpBuilder& builder, tosa::LogOp log) {
  auto x = log.getInput1();
  return createOp<tosa::ReciprocalOp>(builder, x.getType(), x);
}

template <>
Value TosaAddRule::getLhsDerivative(OpBuilder& builder, tosa::AddOp add) {
  return ones(builder, add);
}

template <>
Value TosaAddRule::getRhsDerivative(OpBuilder& builder, tosa::AddOp add) {
  return ones(builder, add);
}

template <>
Value TosaSubRule::getLhsDerivative(OpBuilder& builder, tosa::SubOp sub) {
  return ones(builder, sub);
}

template <>
Value TosaSubRule::getRhsDerivative(OpBuilder& builder, tosa::SubOp sub) {
  auto os = ones(builder, sub);
  return createOp<tosa::NegateOp>(builder, os.getType(), os);
}

template <>
Value TosaMulRule::getLhsDerivative(OpBuilder& builder, tosa::MulOp mul) {
  return mul.getInput2();
}

template <>
Value TosaMulRule::getRhsDerivative(OpBuilder& builder, tosa::MulOp mul) {
  return mul.getInput1();
}

}  // namespace mlir::autodiff
