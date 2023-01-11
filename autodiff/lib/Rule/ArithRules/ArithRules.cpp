#include "ArithRules.hpp"

#include "AddRules.cpp"
#include "DivRules.cpp"
#include "MaxRules.cpp"
#include "MinRules.cpp"
#include "MulRules.cpp"
#include "NegRules.cpp"
#include "SubRules.cpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir::autodiff {

ValueRange getArithGradients(Operation* op, Value grad) {
  if (isa<arith::AddIOp>(op))
    return getGradients<ArithmeticAddIRule>(op, grad);

  else if (isa<arith::AddFOp>(op))
    return getGradients<ArithmeticAddFRule>(op, grad);

  else if (isa<arith::SubIOp>(op))
    return getGradients<ArithmeticSubIRule>(op, grad);

  else if (isa<arith::SubFOp>(op))
    return getGradients<ArithmeticSubFRule>(op, grad);

  else if (isa<arith::MulIOp>(op))
    return getGradients<ArithmeticMulIRule>(op, grad);

  else if (isa<arith::MulFOp>(op))
    return getGradients<ArithmeticMulFRule>(op, grad);

  else if (isa<arith::DivFOp>(op))
    return getGradients<ArithmeticDivFRule>(op, grad);

  else if (isa<arith::MaxFOp>(op))
    return getGradients<ArithmeticMaxFRule>(op, grad);

  else if (isa<arith::MaxSIOp>(op))
    return getGradients<ArithmeticMaxSIRule>(op, grad);

  else if (isa<arith::MaxUIOp>(op))
    return getGradients<ArithmeticMaxUIRule>(op, grad);

  else if (isa<arith::MinFOp>(op))
    return getGradients<ArithmeticMinFRule>(op, grad);

  else if (isa<arith::MinSIOp>(op))
    return getGradients<ArithmeticMinSIRule>(op, grad);

  else if (isa<arith::MinUIOp>(op))
    return getGradients<ArithmeticMinUIRule>(op, grad);

  else if (isa<arith::NegFOp>(op))
    return getGradients<ArithmeticNegFRule>(op, grad);

  else
    assert(false && "Unsupported `arith` operation detected");
}

Value getArithGradient(Operation* op, Value grad, Value input) {
  if (isa<arith::AddIOp>(op))
    return getGradient<ArithmeticAddIRule>(op, grad, input);

  else if (isa<arith::AddFOp>(op))
    return getGradient<ArithmeticAddFRule>(op, grad, input);

  else if (isa<arith::SubIOp>(op))
    return getGradient<ArithmeticSubIRule>(op, grad, input);

  else if (isa<arith::SubFOp>(op))
    return getGradient<ArithmeticSubFRule>(op, grad, input);

  else if (isa<arith::MulIOp>(op))
    return getGradient<ArithmeticMulIRule>(op, grad, input);

  else if (isa<arith::MulFOp>(op))
    return getGradient<ArithmeticMulFRule>(op, grad, input);

  else if (isa<arith::DivFOp>(op))
    return getGradient<ArithmeticDivFRule>(op, grad, input);

  else if (isa<arith::MaxFOp>(op))
    return getGradient<ArithmeticMaxFRule>(op, grad, input);

  else if (isa<arith::MaxSIOp>(op))
    return getGradient<ArithmeticMaxSIRule>(op, grad, input);

  else if (isa<arith::MaxUIOp>(op))
    return getGradient<ArithmeticMaxUIRule>(op, grad, input);

  else if (isa<arith::MinFOp>(op))
    return getGradient<ArithmeticMinFRule>(op, grad, input);

  else if (isa<arith::MinSIOp>(op))
    return getGradient<ArithmeticMinSIRule>(op, grad, input);

  else if (isa<arith::MinUIOp>(op))
    return getGradient<ArithmeticMinUIRule>(op, grad, input);

  else if (isa<arith::NegFOp>(op))
    return getGradient<ArithmeticNegFRule>(op, grad, input);

  else
    assert(false && "Unsupported `arith` operation detected");
}

}  // namespace mlir::autodiff
