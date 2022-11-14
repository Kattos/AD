#include "ArithRules.hpp"

#include "AddRules.cpp"
#include "DivRules.cpp"
#include "MulRules.cpp"
#include "SubRules.cpp"

namespace mlir::autodiff {

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

  else
    assert(false && "Unsupported `arith` operation detected");
}

}  // namespace mlir::autodiff
