#include "ArithRules.hpp"

namespace mlir::autodiff {

template <>
Value ArithmeticNegFRule::getInputDerivative(OpBuilder& builder,
                                        arith::NegFOp negf) {
  return createOp<arith::NegFOp>(builder, ones(builder, negf));
}

}  // namespace mlir::autodiff
