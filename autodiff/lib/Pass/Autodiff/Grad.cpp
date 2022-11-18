#include "Dialect/AD/IR/AD.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::autodiff {

class GradPass : public GradPassBase<GradPass> {
  void runOnOperation() override {}
};

std::unique_ptr<Pass> createADGradPass() {
  return std::make_unique<GradPass>();
}

}  // namespace mlir::autodiff
