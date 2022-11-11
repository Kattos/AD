#ifndef AD_CONVERSION_ADTOTOSA_H
#define AD_CONVERSION_ADTOTOSA_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {

#define GEN_PASS_DECL_ADTOTOSA
#define GEN_PASS_DEF_ADTOTOSA
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createADToTosa();

}  // namespace mlir::autodiff

#endif  // AD_CONVERSION_ADTOTOSA_H
