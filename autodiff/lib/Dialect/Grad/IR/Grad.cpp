#include "Dialect/Grad/IR/Grad.hpp"

#define GET_OP_CLASSES
#include "Dialect/Grad/IR/Grad.cpp.inc"

namespace mlir::autodiff::grad {

bool isTypeLegal(Type input, Type grad) {
  if (isa<FloatType>(input) && isa<FloatType>(grad)) {
    return true;
  }

  if (isa<IntegerType>(input) && isa<IntegerType>(grad)) {
    return true;
  }

  if (isa<TensorType>(input) && isa<TensorType>(grad)) {
    return true;
  }

  return false;
}

bool isOpTypeLegal(Operation *op) {
  assert(3 == op->getNumOperands() && 2 == op->getNumResults() &&
         "Operation is not grad binary op");

  auto lhs = op->getOperand(0).getType();
  auto dLhs = op->getResult(0).getType();

  auto rhs = op->getOperand(1).getType();
  auto dRhs = op->getResult(1).getType();

  return isTypeLegal(lhs, dLhs) && isTypeLegal(rhs, dRhs);
}

LogicalResult AddOp::verify() {
  return isOpTypeLegal(*this) ? success() : failure();
}

}  // namespace mlir::autodiff::grad
