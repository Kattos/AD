#include "Rule/Rules.hpp"

#include "ArithRules/ArithRules.hpp"
#include "MathRules/MathRules.hpp"
#include "TosaRules/TosaRules.hpp"

namespace mlir::autodiff {

ValueRange getGradients(Operation* op, Value grad) {
  if (isIn<tosa::TosaDialect>(op))
    return getTosaGradients(op, grad);

  else if (isIn<math::MathDialect>(op))
    return getMathGradients(op, grad);

  else if (isIn<arith::ArithmeticDialect>(op))
    return getArithGradients(op, grad);

  return {};
}

Value getGradient(Operation* op, Value grad, Value input) {
  if (isIn<tosa::TosaDialect>(op))
    return getTosaGradient(op, grad, input);

  else if (isIn<math::MathDialect>(op))
    return getMathGradient(op, grad, input);

  else if (isIn<arith::ArithmeticDialect>(op))
    return getArithGradient(op, grad, input);

  return nullptr;
}

}  // namespace mlir::autodiff
