
#include "ADUtils.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::autodiff {

// simple way to get ones or zeros
Value ones(OpBuilder& builder, Value value) {
  return createOp<ad::OneslikeOp>(builder, value);
}

Value zeros(OpBuilder& builder, Value value) {
  return createOp<ad::ZeroslikeOp>(builder, value);
}

Value sum(OpBuilder& builder, Value lhs, Value rhs) {
  if (!lhs && !rhs) {
    return nullptr;
  }
  if (!lhs) {
    return rhs;
  }
  if (!rhs) {
    return lhs;
  }
  return createOp<tosa::AddOp>(builder, lhs.getType(), lhs, rhs);
}

Value product(OpBuilder& builder, Value lhs, Value rhs) {
  if (!lhs || !rhs) {
    return nullptr;
  }
  return createOp<tosa::MulOp>(builder, lhs.getType(), lhs, rhs);
}

// get operation by value or get value by operation
Operation* getRelatedOperation(Value value) {
  return isa<OpResult>(value) ? value.getDefiningOp() : nullptr;
}

Value getRelatedValue(Operation* op) {
  return op->getNumResults() == 1 ? op->getResult(0) : nullptr;
}

}  // namespace mlir::autodiff