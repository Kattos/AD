#include "MathRules.hpp"

namespace mlir::autodiff {

template <>
Value MathExpRule::getInputDerivative(OpBuilder& builder, math::ExpOp exp) {
  return exp;
}

}  // namespace mlir::autodiff
