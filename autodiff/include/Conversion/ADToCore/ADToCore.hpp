#ifndef AD_CONVERSION_ADTOCORE_H
#define AD_CONVERSION_ADTOCORE_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {

#define GEN_PASS_DECL_ADTOCORE
#define GEN_PASS_DEF_ADTOCORE
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createADToCore();

}  // namespace mlir::autodiff

#endif  // AD_CONVERSION_ADTOCORE_H
