#ifndef AD_CONVERSION_LNIALGEXTCONVERSION_H
#define AD_CONVERSION_LNIALGEXTCONVERSION_H

#include "Dialect/LinalgExt/IR/LinalgExt.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {

#define GEN_PASS_DECL_ALLOCTENSORTOINITTENSOR
#define GEN_PASS_DEF_ALLOCTENSORTOINITTENSOR
#define GEN_PASS_DECL_INITTENSORTOALLOCTENSOR
#define GEN_PASS_DEF_INITTENSORTOALLOCTENSOR
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createInitTensorToAllocTensor();
std::unique_ptr<Pass> createAllocTensorToInitTensor();

}  // namespace autodiff
}  // namespace mlir

#endif  // AD_CONVERSION_LNIALGEXTCONVERSION_H
