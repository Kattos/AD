#include "ArithRules.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace mlir::autodiff {

template <>
Value ArithSubIRule::getLhsDerivative(OpBuilder& builder, arith::SubIOp subi) {
  return ones(builder, subi);
}

template <>
Value ArithSubIRule::getRhsDerivative(OpBuilder& builder, arith::SubIOp subi) {
  auto width = subi.getType().cast<IntegerType>().getWidth();
  return createOp<arith::ConstantIntOp>(builder, -1, width);
}

template <>
Value ArithSubFRule::getLhsDerivative(OpBuilder& builder, arith::SubFOp subf) {
  return ones(builder, subf);
}

template <>
Value ArithSubFRule::getRhsDerivative(OpBuilder& builder, arith::SubFOp subf) {
  auto one = ones(builder, subf);
  return createOp<arith::NegFOp>(builder, one);
}

}  // namespace mlir::autodiff
