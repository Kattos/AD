#ifndef AD_CONVERSION_GRADTOCORE_H
#define AD_CONVERSION_GRADTOCORE_H

#include "Dialect/Grad/IR/GradDialect.hpp"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {

#define GEN_PASS_DECL_GRADTOCORE
#define GEN_PASS_DEF_GRADTOCORE
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createGradToCore();

}  // namespace mlir::autodiff

#endif  // AD_CONVERSION_GRADTOCORE_H
