#ifndef AD_CONVERSION_GRADTOCORE_H
#define AD_CONVERSION_GRADTOCORE_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {

#define GEN_PASS_DECL_GRADTOCORE
#define GEN_PASS_DEF_GRADTOCORE
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createGradToCore();

}  // namespace mlir::autodiff

#endif  // AD_CONVERSION_GRADTOCORE_H
