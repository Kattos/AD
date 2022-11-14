// Main entry function for autodiff-opt and derived binaries.
//
// Based on mlir-opt but registers the passes and dialects we care about.

#include "InitAD.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::autodiff::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::autodiff::registerAllDialects(registry);

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Autodiff modular optimizer driver\n", registry));
}
