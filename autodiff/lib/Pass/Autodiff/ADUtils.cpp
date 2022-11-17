
#include "ADUtils.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

  // TODO: support different input types
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();

  if (isa<TensorType>(lhsType) || isa<TensorType>(rhsType)) {
    return createOp<tosa::AddOp>(builder, lhs.getType(), lhs, rhs);
  } else if (isa<FloatType>(lhsType) || isa<FloatType>(rhsType)) {
    return createOp<arith::AddFOp>(builder, lhs, rhs);
  } else if (isa<IntegerType>(lhsType) || isa<IntegerType>(rhsType)) {
    return createOp<arith::AddIOp>(builder, lhs, rhs);
  }

  return nullptr;
}

Value product(OpBuilder& builder, Value lhs, Value rhs) {
  if (!lhs || !rhs) {
    return nullptr;
  }

  // TODO: support different input types
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();

  if (isa<TensorType>(lhsType) || isa<TensorType>(rhsType)) {
    auto shift = builder.getI32IntegerAttr(0);
    return createOp<tosa::MulOp>(builder, lhs.getType(), lhs, rhs, shift);
  } else if (isa<FloatType>(lhsType) || isa<FloatType>(rhsType)) {
    return createOp<arith::MulFOp>(builder, lhs, rhs);
  } else if (isa<IntegerType>(lhsType) || isa<IntegerType>(rhsType)) {
    return createOp<arith::MulIOp>(builder, lhs, rhs);
  }

  return nullptr;
}

// get operation by value or get value by operation
Operation* getRelatedOperation(Value value) {
  return isa<OpResult>(value) ? value.getDefiningOp() : nullptr;
}

Value getRelatedValue(Operation* op) {
  return op->getNumResults() == 1 ? op->getResult(0) : nullptr;
}

Value cmpF(OpBuilder& builder, Value lhs, Value rhs,
           arith::CmpFPredicate predicate) {
  return createOp<arith::CmpFOp>(builder, predicate, lhs, rhs);
}

Value cmpF(OpBuilder& builder, Operation* op, arith::CmpFPredicate predicate) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  return cmpF(builder, lhs, rhs, predicate);
}

Value cmpI(OpBuilder& builder, Value lhs, Value rhs,
           arith::CmpIPredicate predicate) {
  return createOp<arith::CmpIOp>(builder, predicate, lhs, rhs);
}

Value cmpI(OpBuilder& builder, Operation* op, arith::CmpIPredicate predicate) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  return cmpI(builder, lhs, rhs, predicate);
}

// TODO: implement reduce function
Value reduce(OpBuilder& builder, Value larger, Value smaller) {
  auto largerType = larger.getType().cast<TensorType>();
  auto smallerType = smaller.getType().cast<TensorType>();
  assert(largerType && smallerType && "Support tensor ruduce only");

  return nullptr;
}

}  // namespace mlir::autodiff
