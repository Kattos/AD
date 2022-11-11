#ifndef SIMPLIFY_PASSES_H
#define SIMPLIFY_PASSES_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {
std::unique_ptr<Pass> createADSimplifyPass();
}  // namespace mlir::autodiff

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "Pass/Simplify/Passes.hpp.inc"

#endif  // SIMPLIFY_PASSES_H