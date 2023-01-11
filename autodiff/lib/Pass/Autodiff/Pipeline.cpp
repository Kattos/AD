// #include "Conversion/ADToCore/ADToCore.hpp"
// #include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
// #include "Conversion/GradToCore/GradToCore.hpp"
#include "Conversion/Conversion.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::autodiff {

void buildAutodiffPipeline(OpPassManager& passManager) {
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

void builderTosaToLinalgPipeline(OpPassManager& passManager) {
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  passManager.addNestedPass<func::FuncOp>(
      createLinalgElementwiseOpFusionPass());
}

void registerTosaToLinalgPipeline() {
  PassPipelineRegistration<> tosaToLinalg(
      "tosa-to-linalg",
      "Transform tosa ops to linalg ops and fuse elementwise ops",
      [](OpPassManager& passManager) {
        builderTosaToLinalgPipeline(passManager);
      });
}

}  // namespace mlir::autodiff
