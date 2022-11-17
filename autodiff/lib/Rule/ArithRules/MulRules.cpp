#include "ArithRules.hpp"

namespace mlir::autodiff {

template <>
Value ArithMulIRule::getLhsDerivative(OpBuilder& builder, arith::MulIOp muli) {
  return muli.getRhs();
}

template <>
Value ArithMulIRule::getRhsDerivative(OpBuilder& builder, arith::MulIOp muli) {
  return muli.getLhs();
}

template <>
Value ArithMulFRule::getLhsDerivative(OpBuilder& builder, arith::MulFOp mulf) {
  return mulf.getRhs();
}

template <>
Value ArithMulFRule::getRhsDerivative(OpBuilder& builder, arith::MulFOp mulf) {
  return mulf.getLhs();
}

}  // namespace mlir::autodiff
