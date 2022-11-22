#include "Dialect/AD/IR/AD.hpp"

#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "Dialect/AD/IR/AD.cpp.inc"

namespace mlir {
namespace OpTrait {
namespace impl {

LogicalResult verifyResultsAreScalarTensorLike(Operation* op) {
  for (auto resultType : op->getResultTypes()) {
    if (!isa<TensorType>(resultType)) {
      return failure();
    }

    auto shape = resultType.cast<TensorType>().getShape();

    for (auto dim : shape) {
      if (dim != 1) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult verifySameToAndResultType(Operation* op) {
  if (failed(verifyNOperands(op, 2))) {
    return failure();
  }

  if (failed(verifyOneResult(op))) {
    return failure();
  }

  auto toType = op->getOperand(1).getType();
  auto resType = op->getResult(0).getType();

  return success(toType == resType);
}

}  // namespace impl
}  // namespace OpTrait

namespace autodiff {
namespace ad {

void BroadcastOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Value from, Value to) {
  return BroadcastOp::build(odsBuilder, odsState, to.getType(), from, to);
}

void ToTensorOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       Value input) {
  auto inputType = input.getType();

  if (isa<TensorType>(inputType)) {
    return build(odsBuilder, odsState, inputType, input);
  }

  auto type = RankedTensorType::get({}, getElementTypeOrSelf(inputType));
  return build(odsBuilder, odsState, type, input);
}

}  // namespace ad
}  // namespace autodiff
}  // namespace mlir
