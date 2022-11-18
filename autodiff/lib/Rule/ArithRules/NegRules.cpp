#include "ArithRules.hpp"

namespace mlir::autodiff {

template <>
Value ArithNegFRule::getInputDerivative(OpBuilder& builder,
                                        arith::NegFOp negf) {
  return createOp<arith::NegFOp>(builder, ones(builder, negf));
}

}  // namespace mlir::autodiff
