#include "Dialect/AD/IR/AD.hpp"

#define GET_OP_CLASSES
#include "Dialect/AD/IR/AD.cpp.inc"

namespace mlir::autodiff::ad {
void ZeroslikeOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Value input) {
  odsState.addOperands(input);
  odsState.addTypes(input.getType());
}

void OneslikeOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       Value input) {
  odsState.addOperands(input);
  odsState.addTypes(input.getType());
}
}  // namespace mlir::autodiff::ad