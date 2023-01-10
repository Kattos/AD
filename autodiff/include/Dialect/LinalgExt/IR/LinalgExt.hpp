#ifndef LINALGEXT_DIALECT_H
#define LINALGEXT_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "mlir/IR/Dialect.h"
#include "Dialect/LinalgExt/IR/LinalgExtDialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "Dialect/LinalgExt/IR/LinalgExt.h.inc"

#endif  // LINALGEXT_H
