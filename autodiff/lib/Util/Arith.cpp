#include "Util/Arith.hpp"

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

Value add(Value lhs, Value rhs, OpBuilder& builder) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>()) {
    return builder.create<AddIOp>(builder.getUnknownLoc(), lhs, rhs);
  }
  return builder.create<AddFOp>(builder.getUnknownLoc(), lhs, rhs);
}

Value mul(Value lhs, Value rhs, OpBuilder& builder) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>()) {
    return builder.create<MulIOp>(builder.getUnknownLoc(), lhs, rhs);
  }
  return builder.create<MulFOp>(builder.getUnknownLoc(), lhs, rhs);
}

}  // namespace arith
}  // namespace util
}  // namespace autodiff
}  // namespace mlir
