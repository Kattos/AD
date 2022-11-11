#include "Rules.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::autodiff {

ValueRange getGradients(Operation* op, Value grad) {
  if (isa<tosa::ExpOp>(op))
    return getGradients<ExpOpRule>(op, grad);
  else if (isa<tosa::LogOp>(op))
    return getGradients<LogOpRule>(op, grad);
  else if (isa<tosa::AddOp>(op))
    return getGradients<AddOpRule>(op, grad);
  else if (isa<tosa::SubOp>(op))
    return getGradients<SubOpRule>(op, grad);
  else if (isa<tosa::MulOp>(op))
    return getGradients<MulOpRule>(op, grad);
  else
    assert(false && "Unsupported operation detected");
}

Value getGradient(Operation* op, Value grad, Value input) {
  if (isa<tosa::ExpOp>(op))
    return getGradient<ExpOpRule>(op, grad, input);
  else if (isa<tosa::LogOp>(op))
    return getGradient<LogOpRule>(op, grad, input);
  else if (isa<tosa::AddOp>(op))
    return getGradient<AddOpRule>(op, grad, input);
  else if (isa<tosa::SubOp>(op))
    return getGradient<SubOpRule>(op, grad, input);
  else if (isa<tosa::MulOp>(op))
    return getGradient<MulOpRule>(op, grad, input);
  else
    assert(false && "Unsupported operation detected");
}

template <>
Value ExpOpRule::getInputDerivative(OpBuilder& builder, tosa::ExpOp exp) {
  auto x = exp.getInput1();
  return product(builder, exp, x);
}

template <>
Value LogOpRule::getInputDerivative(OpBuilder& builder, tosa::LogOp log) {
  auto x = log.getInput1();
  return createOp<tosa::ReciprocalOp>(builder, x.getType(), x);
}

template <>
Value AddOpRule::getLhsDerivative(OpBuilder& builder, tosa::AddOp add) {
  return ones(builder, add);
}

template <>
Value AddOpRule::getRhsDerivative(OpBuilder& builder, tosa::AddOp add) {
  return ones(builder, add);
}

template <>
Value SubOpRule::getLhsDerivative(OpBuilder& builder, tosa::SubOp sub) {
  return ones(builder, sub);
}

template <>
Value SubOpRule::getRhsDerivative(OpBuilder& builder, tosa::SubOp sub) {
  auto os = ones(builder, sub);
  return createOp<tosa::NegateOp>(builder, os.getType(), os);
}

template <>
Value MulOpRule::getLhsDerivative(OpBuilder& builder, tosa::MulOp mul) {
  return mul.getInput2();
}

template <>
Value MulOpRule::getRhsDerivative(OpBuilder& builder, tosa::MulOp mul) {
  return mul.getInput1();
}

}  // namespace mlir::autodiff