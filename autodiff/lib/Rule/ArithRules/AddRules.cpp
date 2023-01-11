#include "ArithRules.hpp"

namespace mlir::autodiff {

template <>
Value ArithmeticAddIRule::getLhsDerivative(OpBuilder& builder, arith::AddIOp addi) {
  return ones(builder, addi);
}

template <>
Value ArithmeticAddIRule::getRhsDerivative(OpBuilder& builder, arith::AddIOp addi) {
  return ones(builder, addi);
}

template <>
Value ArithmeticAddFRule::getLhsDerivative(OpBuilder& builder, arith::AddFOp addf) {
  return ones(builder, addf);
}

template <>
Value ArithmeticAddFRule::getRhsDerivative(OpBuilder& builder, arith::AddFOp addf) {
  return ones(builder, addf);
}

}  // namespace mlir::autodiff
