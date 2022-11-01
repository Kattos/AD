// clang-format off
#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"

#include "Dialect/AD/IR/ADDialect.cpp.inc"
// clang-format on

void mlir::autodiff::ad::ADDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/AD/IR/AD.cpp.inc"
      >();
}
