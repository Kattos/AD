#include "TosaRules.hpp"

namespace mlir::autodiff {

template <>
Value TosaNegateRule::getInputDerivative(OpBuilder& builder,
                                         tosa::NegateOp negate) {
  return createOp<tosa::NegateOp>(builder, negate.getType(),
                                  ones(builder, negate));
}

}  // namespace mlir::autodiff
