#include "Util/Bufferization.hpp"

namespace mlir {
namespace autodiff {
namespace util {
namespace bufferization {

using namespace mlir::bufferization;

Value alloc(Value value, OpBuilder& builder) {
  static SmallVector<Value, 0> dynamicSizes;
  static auto operandSegmentSizes = builder.getNamedAttr(
      "operand_segment_sizes", builder.getI32ArrayAttr({0, 0}));

  return builder.create<AllocTensorOp>(builder.getUnknownLoc(), value.getType(),
                                       dynamicSizes, operandSegmentSizes);
}

}  // namespace bufferization
}  // namespace util
}  // namespace autodiff
}  // namespace mlir