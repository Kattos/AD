#include "Rules.hpp"

#include "ADUtils.hpp"
#include "MathRules.hpp"
#include "TosaRules.hpp"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::autodiff {

ValueRange getGradients(Operation* op, Value grad) {
  if (isIn<tosa::TosaDialect>(op)) {
    return getTosaGradients(op, grad);
  } else if (isIn<math::MathDialect>(op)) {
    return getMathGradients(op, grad);
  }
  return {};
}

Value getGradient(Operation* op, Value grad, Value input) {
  if (isIn<tosa::TosaDialect>(op)) {
    return getTosaGradient(op, grad, input);
  } else if (isIn<math::MathDialect>(op)) {
    return getMathGradient(op, grad, input);
  }
  return nullptr;
}

}  // namespace mlir::autodiff
