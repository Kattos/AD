#include "Rule/Utils.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir::autodiff {

Value cmpF(OpBuilder& builder, Value lhs, Value rhs,
           arith::CmpFPredicate predicate);
Value cmpF(OpBuilder& builder, Operation* op, arith::CmpFPredicate predicate);
Value cmpI(OpBuilder& builder, Value lhs, Value rhs,
           arith::CmpIPredicate predicate);
Value cmpI(OpBuilder& builder, Operation* op, arith::CmpIPredicate predicate);

}  // namespace mlir::autodiff
