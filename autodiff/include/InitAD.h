// This file defines a helper to add passes and dialects to the global registry.

#ifndef INIT_AD_H
#define INIT_AD_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace autodiff {
void registerAllPasses();
void registerAllDialects(DialectRegistry &registry);
}  // namespace autodiff
}  // namespace mlir

#endif  // INIT_AD_H
