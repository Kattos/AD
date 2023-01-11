#include "ArithRules.hpp"

namespace mlir::autodiff {

template <>
Value ArithmeticMulIRule::getLhsDerivative(OpBuilder& builder, arith::MulIOp muli) {
  return muli.getRhs();
}

template <>
Value ArithmeticMulIRule::getRhsDerivative(OpBuilder& builder, arith::MulIOp muli) {
  return muli.getLhs();
}

template <>
Value ArithmeticMulFRule::getLhsDerivative(OpBuilder& builder, arith::MulFOp mulf) {
  return mulf.getRhs();
}

template <>
Value ArithmeticMulFRule::getRhsDerivative(OpBuilder& builder, arith::MulFOp mulf) {
  return mulf.getLhs();
}

}  // namespace mlir::autodiff
