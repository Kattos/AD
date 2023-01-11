#include "Rule/Utils.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir::autodiff {

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

}  // namespace mlir::autodiff
