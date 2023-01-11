#include "Dialect/LinalgExt/IR/LinalgExt.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Dialect/LinalgExt/IR/LinalgExt.cpp.inc"
#include "Dialect/LinalgExt/IR/LinalgExtDialect.cpp.inc"

namespace mlir {
namespace autodiff {
namespace linalgext {

void LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/LinalgExt/IR/LinalgExt.cpp.inc"
      >();
}

void InitTensorOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         ArrayRef<int64_t> shape, Type elemtype) {
  auto type = RankedTensorType::get(shape, elemtype);
  auto attr = odsBuilder.getI64ArrayAttr(shape);
  return build(odsBuilder, odsState, type, attr);
}

LogicalResult InitTensorOp::verify() {
  auto tensorShape = getType().getShape();
  auto attrShape = shape().getValue();

  if (tensorShape.size() != attrShape.size()) {
    emitError() << "Shape of tensor conflicts with the given attribute\n";
  }

  for (auto [ts, as] : llvm::zip(tensorShape, attrShape)) {
    if (!as.isa<IntegerAttr>() || ts != as.cast<IntegerAttr>().getInt()) {
      emitError() << "Shape of tensor conflicts with the given attribute\n";
      return failure();
    }
  }

  return success();
}

}  // namespace linalgext
}  // namespace autodiff
}  // namespace mlir
