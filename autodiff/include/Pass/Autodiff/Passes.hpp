#ifndef AUTODIFF_PASSES_H
#define AUTODIFF_PASSES_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {
std::unique_ptr<Pass> createADPreprocessPass();
std::unique_ptr<Pass> createADNaivePass();
std::unique_ptr<Pass> createADExperimentalPass();
}  // namespace autodiff
}  // namespace mlir

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "Pass/Autodiff/Passes.hpp.inc"

#endif  // AUTODIFF_PASSES_H