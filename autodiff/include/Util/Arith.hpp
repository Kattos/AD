#ifndef AD_UTIL_ARITH_HPP
#define AD_UTIL_ARITH_HPP

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace autodiff {
namespace util {
namespace arith {

Value constant(double value, OpBuilder& builder, Type type = nullptr);
Value add(Value lhs, Value rhs, OpBuilder& builder);
Value mul(Value lhs, Value rhs, OpBuilder& builder);
Value sub(Value lhs, Value rhs, OpBuilder& builder);
Value div(Value lhs, Value rhs, OpBuilder& builder);

}  // namespace arith
}  // namespace util
}  // namespace autodiff
}  // namespace mlir

#endif  // AD_UTIL_ARITH_HPP
