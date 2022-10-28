// clang-format off
#include "Dialect/ADTosa/IR/ADTosa.h"
#include "Dialect/ADTosa/IR/ADTosaDialect.h"

#include "Dialect/ADTosa/IR/ADTosaDialect.cpp.inc"
// clang-format on

void mlir::autodiff::ad_tosa::ADTosaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ADTosa/IR/ADTosa.cpp.inc"
      >();
}
