
#include "Rule/Utils.hpp"

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

LogicalResult notNull(Value value) { return success(value != nullptr); }

int64_t counter() {
  static int64_t index = 0;
  return ++index;
}

Value reduce(OpBuilder& builder, Value from, Value to) {
  auto fromType = from.getType();
  auto toType = to.getType();

  assert(isa<ShapedType>(fromType) && isa<ShapedType>(toType));

  auto fromShape = fromType.cast<ShapedType>().getShape();
  auto toShape = toType.cast<ShapedType>().getShape();

  if (fromShape == toShape || fromShape.size() < toShape.size()) {
    return from;
  }

  auto elemType = fromType.cast<ShapedType>().getElementType();

  auto fromVec = fromShape.vec();
  auto toVec = toShape.vec();
  while (fromVec.size() > toVec.size()) {
    toVec.emplace_back(1);
  }

  for (size_t i = 0; i < fromShape.size(); ++i) {
    auto fromDim = fromVec[i];
    auto toDim = toVec[i];

    if (fromDim == toDim) {
      continue;
    }

    assert(toDim == 1 && "Cannot reduce along non-singleton dimension");

    fromVec[i] = 1;
    auto type = RankedTensorType::get(fromVec, elemType);

    // reduce-able
    auto axis = builder.getI64IntegerAttr(i);
    from = createOp<tosa::ReduceSumOp>(builder, type, from, axis);
  }

  auto attr = builder.getI64ArrayAttr(toShape);
  // reshape
  return createOp<tosa::ReshapeOp>(builder, to.getType(), from, attr);
}

Value broadcast(OpBuilder& builder, Value from, Value to) {
  auto fromType = from.getType();
  auto toType = to.getType();

  assert(isa<ShapedType>(fromType) && isa<ShapedType>(toType));

  auto fromShape = fromType.cast<ShapedType>().getShape();
  auto toShape = toType.cast<ShapedType>().getShape();

  if (fromShape == toShape) {
    return from;
  }

  auto attr = builder.getI64ArrayAttr(toShape);
  return createOp<tosa::ReshapeOp>(builder, to.getType(), from, attr);
}

}  // namespace mlir::autodiff
