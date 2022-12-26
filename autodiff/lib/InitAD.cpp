#include "InitAD.h"

#include "Conversion/Conversion.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Pass/Simplify/Passes.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace autodiff {

void registerAllPasses() {
  // TODO: register all passes needed here
  registerAutodiffPasses();
  registerConversionPasses();
  // registerTosaConversionPasses();
  // registerSimplifyPasses();

  registerAutodiffPipeline();
}

void registerAllDialects(DialectRegistry &registry) {
  // TODO: register all dialects needed here
  registry.insert<func::FuncDialect>();
  registry.insert<tosa::TosaDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<ml_program::MLProgramDialect>();

  registry.insert<ad::ADDialect>();
  registry.insert<grad::GradDialect>();
}

}  // namespace autodiff
}  // namespace mlir
