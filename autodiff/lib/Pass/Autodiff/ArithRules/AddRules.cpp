#include "ArithRules.hpp"

namespace mlir::autodiff {

template <>
Value ArithAddIRule::getLhsDerivative(OpBuilder& builder, arith::AddIOp addi) {
  return ones(builder, addi);
}

template <>
Value ArithAddIRule::getRhsDerivative(OpBuilder& builder, arith::AddIOp addi) {
  return ones(builder, addi);
}

template <>
Value ArithAddFRule::getLhsDerivative(OpBuilder& builder, arith::AddFOp addf) {
  return ones(builder, addf);
}

template <>
Value ArithAddFRule::getRhsDerivative(OpBuilder& builder, arith::AddFOp addf) {
  return ones(builder, addf);
}

}  // namespace mlir::autodiff
