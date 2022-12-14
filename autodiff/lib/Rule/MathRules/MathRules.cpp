#include "MathRules.hpp"

#include "AbsRules.cpp"
#include "ExpRules.cpp"
#include "LogRules.cpp"
#include "RsqrtRules.cpp"
#include "TanhRules.cpp"

namespace mlir::autodiff {

ValueRange getMathGradients(Operation* op, Value grad) {
  if (isa<math::AbsFOp>(op))
    return getGradients<MathAbsFRule>(op, grad);

  else if (isa<math::AbsIOp>(op))
    return getGradients<MathAbsIRule>(op, grad);

  else if (isa<math::LogOp>(op))
    return getGradients<MathLogRule>(op, grad);

  else if (isa<math::Log10Op>(op))
    return getGradients<MathLog10Rule>(op, grad);

  else if (isa<math::Log2Op>(op))
    return getGradients<MathLog2Rule>(op, grad);

  else if (isa<math::Log1pOp>(op))
    return getGradients<MathLog1pRule>(op, grad);

  else if (isa<math::RsqrtOp>(op))
    return getGradients<MathRsqrtRule>(op, grad);

  else if (isa<math::ExpOp>(op))
    return getGradients<MathExpRule>(op, grad);

  else if (isa<math::TanhOp>(op))
    return getGradients<MathTanhRule>(op, grad);

  else
    assert(false && "Unsupported `math` operation detected");
}

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

  else if (isa<math::RsqrtOp>(op))
    return getGradient<MathRsqrtRule>(op, grad, input);

  else if (isa<math::ExpOp>(op))
    return getGradient<MathExpRule>(op, grad, input);

  else if (isa<math::TanhOp>(op))
    return getGradient<MathTanhRule>(op, grad, input);

  else
    assert(false && "Unsupported `math` operation detected");
}

}  // namespace mlir::autodiff
