#include "ArithRules.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"

namespace mlir::autodiff {

template <>
Value ArithmeticDivFRule::getLhsDerivative(OpBuilder& builder, arith::DivFOp divf) {
  auto one = ones(builder, divf);
  return createOp<arith::DivFOp>(builder, one, divf.getRhs());
}

template <>
Value ArithmeticDivFRule::getRhsDerivative(OpBuilder& builder, arith::DivFOp divf) {
  auto rhs = divf.getRhs();

  auto one = ones(builder, divf);

  auto square = createOp<arith::MulFOp>(builder, rhs, rhs);
  auto negate = createOp<arith::NegFOp>(builder, square);

  return createOp<arith::DivFOp>(builder, one, negate);
}

}  // namespace mlir::autodiff
