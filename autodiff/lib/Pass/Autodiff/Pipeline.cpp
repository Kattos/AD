#include "Conversion/ADToCore/ADToCore.hpp"
#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
#include "Conversion/GradToCore/GradToCore.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::autodiff {

void buildAutodiffPipeline(OpPassManager& passManager) {
  // TODO: register real autodiff before all
  passManager.addNestedPass<func::FuncOp>(createADGenGradPass());
  passManager.addNestedPass<func::FuncOp>(createGradAbstractToConcrete());
  passManager.addNestedPass<func::FuncOp>(createGradToCore());
  passManager.addNestedPass<func::FuncOp>(createADToCore());
}

void registerAutodiffPipeline() {
  PassPipelineRegistration<> autodiff(
      "autodiff", "Apply autodiff on original IR",
      [](OpPassManager& passManager) { buildAutodiffPipeline(passManager); });
}

}  // namespace mlir::autodiff
