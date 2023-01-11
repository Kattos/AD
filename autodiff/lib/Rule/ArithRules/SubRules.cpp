#include "ArithRules.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"

namespace mlir::autodiff {

template <>
Value ArithmeticSubIRule::getLhsDerivative(OpBuilder& builder, arith::SubIOp subi) {
  return ones(builder, subi);
}

template <>
Value ArithmeticSubIRule::getRhsDerivative(OpBuilder& builder, arith::SubIOp subi) {
  auto width = subi.getType().cast<IntegerType>().getWidth();
  return createOp<arith::ConstantIntOp>(builder, -1, width);
}

template <>
Value ArithmeticSubFRule::getLhsDerivative(OpBuilder& builder, arith::SubFOp subf) {
  return ones(builder, subf);
}

template <>
Value ArithmeticSubFRule::getRhsDerivative(OpBuilder& builder, arith::SubFOp subf) {
  auto one = ones(builder, subf);
  return createOp<arith::NegFOp>(builder, one);
}

}  // namespace mlir::autodiff
