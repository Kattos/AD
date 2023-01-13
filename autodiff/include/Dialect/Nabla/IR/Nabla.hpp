#ifndef NABLA_DIALECT_HPP
#define NABLA_DIALECT_HPP

#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "mlir/IR/Dialect.h"
#include "Dialect/Nabla/IR/NablaDialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "Dialect/Nabla/IR/Nabla.h.inc"

#endif  // NABLA_DIALECT_HPP
