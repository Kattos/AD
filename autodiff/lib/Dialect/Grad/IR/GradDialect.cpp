// clang-format off
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"

#include "Dialect/Grad/IR/GradDialect.cpp.inc"
// clang-format on

void mlir::autodiff::grad::GradDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Grad/IR/Grad.cpp.inc"
      >();
}
