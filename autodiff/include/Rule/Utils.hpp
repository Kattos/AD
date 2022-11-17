#ifndef AD_UTILS_HPP
#define AD_UTILS_HPP

#include "Dialect/AD/IR/AD.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::autodiff {
template <typename DialectType>
bool isIn(Operation* op) {
  auto opIn = op->getDialect()->getNamespace();
  auto dialectNamespace = DialectType::getDialectNamespace();
  return opIn == dialectNamespace;
}

template <typename DialectType>
bool isIn(Value value) {
  return isa<OpResult>(value) && isIn<DialectType>(value.getDefiningOp());
}

// create ops without specifying location
template <typename OpTy, typename... Args>
OpTy createOp(OpBuilder& builder, Args... args) {
  return builder.create<OpTy>(builder.getUnknownLoc(),
                              std::forward<Args>(args)...);
}

Value ones(OpBuilder& builder, Value value);
Value zeros(OpBuilder& builder, Value value);
Value sum(OpBuilder& builder, Value lhs, Value rhs);
Value product(OpBuilder& builder, Value lhs, Value rhs);
Operation* getRelatedOperation(Value value);
Value getRelatedValue(Operation* op);

LogicalResult notNull(Value value);

}  // namespace mlir::autodiff

#endif  // AD_UTILS_HPP
