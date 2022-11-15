#include "MathRules.hpp"

#include "AbsRules.cpp"
#include "LogRules.cpp"

namespace mlir::autodiff {

Value getMathGradient(Operation* op, Value grad, Value input) {
  if (isa<math::AbsFOp>(op))
    return getGradient<MathAbsFRule>(op, grad, input);

  else if (isa<math::AbsIOp>(op))
    return getGradient<MathAbsIRule>(op, grad, input);

  else if (isa<math::LogOp>(op))
    return getGradient<MathLogRule>(op, grad, input);

  else if (isa<math::Log10Op>(op))
    return getGradient<MathLog10Rule>(op, grad, input);

  else if (isa<math::Log2Op>(op))
    return getGradient<MathLog2Rule>(op, grad, input);

  else if (isa<math::Log1pOp>(op))
    return getGradient<MathLog1pRule>(op, grad, input);

  else
    assert(false && "Unsupported `math` operation detected");
}

}  // namespace mlir::autodiff
