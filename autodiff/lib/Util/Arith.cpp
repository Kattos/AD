#include "Util/Arith.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace autodiff {
namespace util {
namespace arith {

using namespace mlir::arith;

Value constant(double value, OpBuilder& builder, Type type) {
  if (type == nullptr) {
    type = builder.getF32Type();
  }
  auto attr = builder.getFloatAttr(type, value);
  return builder.create<ConstantOp>(builder.getUnknownLoc(), attr);
}

template <typename IntOp, typename FloatOp>
Value binop(Value lhs, Value rhs, OpBuilder& builder) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>()) {
    return builder.create<IntOp>(builder.getUnknownLoc(), lhs, rhs);
  }
  return builder.create<FloatOp>(builder.getUnknownLoc(), lhs, rhs);
}

Value add(Value lhs, Value rhs, OpBuilder& builder) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  if (isa<IntegerType>(lhsType) && isa<IntegerType>(rhsType)) {
    return builder.create<AddIOp>(builder.getUnknownLoc(), lhs, rhs);
  }
  return builder.create<AddFOp>(builder.getUnknownLoc(), lhs, rhs);
}

Value mul(Value lhs, Value rhs, OpBuilder& builder) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  if (isa<IntegerType>(lhsType) && isa<IntegerType>(rhsType)) {
    return builder.create<MulIOp>(builder.getUnknownLoc(), lhs, rhs);
  }
  return builder.create<MulFOp>(builder.getUnknownLoc(), lhs, rhs);
}

Value sub(Value lhs, Value rhs, OpBuilder& builder) {
  return binop<SubIOp, SubFOp>(lhs, rhs, builder);
}

Value div(Value lhs, Value rhs, OpBuilder& builder) {
  return binop<DivSIOp, DivFOp>(lhs, rhs, builder);
}

}  // namespace arith
}  // namespace util
}  // namespace autodiff
}  // namespace mlir
