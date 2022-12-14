#ifndef AUTODIFF_PASSES_H
#define AUTODIFF_PASSES_H

#include "Dialect/AD/IR/ADDialect.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "Dialect/Nabla/IR/Nabla.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {

std::unique_ptr<Pass> createADGenGradPass();
std::unique_ptr<Pass> createADGradGenericPass();
std::unique_ptr<Pass> createExperimentalPass();

void registerTosaToLinalgPipeline();
void registerAutodiffPipeline();

}  // namespace autodiff
}  // namespace mlir

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "Pass/Autodiff/Passes.hpp.inc"

#endif  // AUTODIFF_PASSES_H
