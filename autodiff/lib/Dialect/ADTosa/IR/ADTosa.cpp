#include "Dialect/ADTosa/IR/ADTosa.hpp"

#define GET_OP_CLASSES
#include "Dialect/ADTosa/IR/ADTosa.cpp.inc"

namespace mlir::autodiff::ad_tosa {
void ZeroslikeOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Value input) {
  odsState.addOperands(input);
  odsState.addTypes(input.getType());
}
}  // namespace mlir::autodiff::ad_tosa