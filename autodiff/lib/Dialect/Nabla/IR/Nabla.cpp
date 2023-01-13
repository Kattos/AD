#include "Dialect/Nabla/IR/Nabla.hpp"

#define GET_OP_CLASSES
#include "Dialect/Nabla/IR/Nabla.cpp.inc"
#include "Dialect/Nabla/IR/NablaDialect.cpp.inc"

namespace mlir {
namespace autodiff {
namespace nabla {

void NablaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Nabla/IR/Nabla.cpp.inc"
      >();
}

}  // namespace nabla
}  // namespace autodiff
}  // namespace mlir