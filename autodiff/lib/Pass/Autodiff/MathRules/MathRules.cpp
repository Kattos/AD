#include "MathRules.hpp"

#include "AbsRules.cpp"

namespace mlir::autodiff {

Value getMathGradient(Operation* op, Value grad, Value input) {
  if (isa<math::AbsFOp>(op))
    return getGradient<MathAbsFRule>(op, grad, input);

  else if (isa<math::AbsIOp>(op))
    return getGradient<MathAbsIRule>(op, grad, input);

  else
    assert(false && "Unsupported `math` operation detected");
}

}  // namespace mlir::autodiff
