#include "InitAD.h"

#include "Conversion/ADToTosa/ADToTosa.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace autodiff {

void registerAllPasses() {
  // TODO: register all passes needed here
  registerAutodiffPasses();
  registerConversionPasses();
}

void registerAllDialects(DialectRegistry &registry) {
  // TODO: register all dialects needed here
  registry.insert<func::FuncDialect>();
  registry.insert<tosa::TosaDialect>();

  registry.insert<ad::ADDialect>();
}

}  // namespace autodiff
}  // namespace mlir
