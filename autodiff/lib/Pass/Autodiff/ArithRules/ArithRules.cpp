#include "ArithRules.hpp"

#include "AddRules.cpp"
#include "DivRules.cpp"
#include "MaxRules.cpp"
#include "MinRules.cpp"
#include "MulRules.cpp"
#include "SubRules.cpp"

namespace mlir::autodiff {

ValueRange getArithGradients(Operation* op, Value grad) {
  if (isa<arith::AddIOp>(op))
    return getGradients<ArithAddIRule>(op, grad);

  else if (isa<arith::AddFOp>(op))
    return getGradients<ArithAddFRule>(op, grad);

  else if (isa<arith::SubIOp>(op))
    return getGradients<ArithSubIRule>(op, grad);

  else if (isa<arith::SubFOp>(op))
    return getGradients<ArithSubFRule>(op, grad);

  else if (isa<arith::MulIOp>(op))
    return getGradients<ArithMulIRule>(op, grad);

  else if (isa<arith::MulFOp>(op))
    return getGradients<ArithMulFRule>(op, grad);

  else if (isa<arith::DivFOp>(op))
    return getGradients<ArithDivFRule>(op, grad);

  else if (isa<arith::MaxFOp>(op))
    return getGradients<ArithMaxFRule>(op, grad);

  else if (isa<arith::MaxSIOp>(op))
    return getGradients<ArithMaxSIRule>(op, grad);

  else if (isa<arith::MaxUIOp>(op))
    return getGradients<ArithMaxUIRule>(op, grad);

  else if (isa<arith::MinFOp>(op))
    return getGradients<ArithMinFRule>(op, grad);

  else if (isa<arith::MinSIOp>(op))
    return getGradients<ArithMinSIRule>(op, grad);

  else if (isa<arith::MinUIOp>(op))
    return getGradients<ArithMinUIRule>(op, grad);

  else
    assert(false && "Unsupported `arith` operation detected");
}

Value getArithGradient(Operation* op, Value grad, Value input) {
  if (isa<arith::AddIOp>(op))
    return getGradient<ArithAddIRule>(op, grad, input);

  else if (isa<arith::AddFOp>(op))
    return getGradient<ArithAddFRule>(op, grad, input);

  else if (isa<arith::SubIOp>(op))
    return getGradient<ArithSubIRule>(op, grad, input);

  else if (isa<arith::SubFOp>(op))
    return getGradient<ArithSubFRule>(op, grad, input);

  else if (isa<arith::MulIOp>(op))
    return getGradient<ArithMulIRule>(op, grad, input);

  else if (isa<arith::MulFOp>(op))
    return getGradient<ArithMulFRule>(op, grad, input);

  else if (isa<arith::DivFOp>(op))
    return getGradient<ArithDivFRule>(op, grad, input);

  else if (isa<arith::MaxFOp>(op))
    return getGradient<ArithMaxFRule>(op, grad, input);

  else if (isa<arith::MaxSIOp>(op))
    return getGradient<ArithMaxSIRule>(op, grad, input);

  else if (isa<arith::MaxUIOp>(op))
    return getGradient<ArithMaxUIRule>(op, grad, input);

  else if (isa<arith::MinFOp>(op))
    return getGradient<ArithMinFRule>(op, grad, input);

  else if (isa<arith::MinSIOp>(op))
    return getGradient<ArithMinSIRule>(op, grad, input);

  else if (isa<arith::MinUIOp>(op))
    return getGradient<ArithMinUIRule>(op, grad, input);

  else
    assert(false && "Unsupported `arith` operation detected");
}

}  // namespace mlir::autodiff
