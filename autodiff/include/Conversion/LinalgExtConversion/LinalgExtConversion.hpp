#ifndef AD_CONVERSION_LNIALGEXTCONVERSION_H
#define AD_CONVERSION_LNIALGEXTCONVERSION_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "Dialect/LinalgExt/IR/LinalgExt.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {

#define GEN_PASS_CLASSES
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createInitTensorToAllocTensor();
std::unique_ptr<Pass> createAllocTensorToInitTensor();

}  // namespace autodiff
}  // namespace mlir

#endif  // AD_CONVERSION_LNIALGEXTCONVERSION_H
