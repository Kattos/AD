#include "InitAD.h"

#include "Conversion/Conversion.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Pass/Simplify/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace autodiff {

void registerAllPasses() {
  // TODO: register all passes needed here
  registerAutodiffPasses();
  registerConversionPasses();
  registerSimplifyPasses();
}

void registerAllDialects(DialectRegistry &registry) {
  // TODO: register all dialects needed here
  registry.insert<func::FuncDialect>();
  registry.insert<tosa::TosaDialect>();

  registry.insert<ad::ADDialect>();
}

}  // namespace autodiff
}  // namespace mlir
