#include "Conversion/ADToCore/ADToCore.hpp"
#include "Conversion/GradToCore/GradToCore.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::autodiff {

void buildAutodiffPipeline(OpPassManager& passManager) {
  // TODO: register real autodiff before all
  // passManager.addNestedPass<func::FuncOp>(createReverseAutodiffPass());

  // convert customized ops to core ops
  passManager.addNestedPass<func::FuncOp>(createGradToCore());
  passManager.addNestedPass<func::FuncOp>(createADToCore());

  // convert iree-illegal ops to iree-legal ops
  passManager.addNestedPass<func::FuncOp>(
      bufferization::createEmptyTensorToAllocTensorPass());
  passManager.addNestedPass<func::FuncOp>(
      bufferization::createBufferizationBufferizePass());
}

void registerAutodiffPipeline() {
  PassPipelineRegistration<> autodiff(
      "autodiff", "Apply autodiff on original IR",
      [](OpPassManager& passManager) { buildAutodiffPipeline(passManager); });
}

}  // namespace mlir::autodiff
