#ifndef AUTODIFF_PASSES_H
#define AUTODIFF_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {
std::unique_ptr<Pass> createDiffPass();
std::unique_ptr<Pass> createADPreprocessPass();
std::unique_ptr<Pass> createADNaivePass();
}  // namespace autodiff
}  // namespace mlir

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "Pass/Autodiff/Passes.hpp.inc"

#endif  // AUTODIFF_PASSES_H