#include "Rules.hpp"

#include "ArithRules/ArithRules.hpp"
#include "MathRules/MathRules.hpp"
#include "TosaRules/TosaRules.hpp"

namespace mlir::autodiff {

Value getGradient(Operation* op, Value grad, Value input) {
  if (isIn<tosa::TosaDialect>(op)) {
    return getTosaGradient(op, grad, input);
  } else if (isIn<math::MathDialect>(op)) {
    return getMathGradient(op, grad, input);
  } else if (isIn<arith::ArithDialect>(op)) {
    return getArithGradient(op, grad, input);
  }
  return nullptr;
}

}  // namespace mlir::autodiff
