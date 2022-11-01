#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::autodiff {

const StringRef DIFF_FUNC_PREFIX = "diff_";

bool isDiffFunction(func::FuncOp funcOp) {
  return funcOp.getSymName().starts_with(DIFF_FUNC_PREFIX);
}

func::FuncOp declareDiffFunction(func::FuncOp funcOp) {
  // insert function at the end
  decltype(auto) block = funcOp->getParentRegion()->getBlocks().back();
  auto builder = OpBuilder::atBlockEnd(&block);

  auto diffOp = dyn_cast<func::FuncOp>(builder.insert(funcOp->clone()));

  // turn name `func` into `diff_func`
  auto funcName = funcOp.getSymName();
  auto diffName = (DIFF_FUNC_PREFIX + funcName).str();

  // dimension of output should be the same as input
  auto funcType = funcOp.getFunctionType();
  auto diffType =
      builder.getFunctionType(funcType.getInputs(), funcType.getInputs());

  diffOp.setSymName(diffName);
  diffOp.setFunctionType(diffType);
  diffOp.setSymVisibilityAttr(builder.getStringAttr("private"));

  // replace func::ReturnOp result
  diffOp->walk([&diffOp](func::ReturnOp returnOp) {
    // return arguments as placeholders
    returnOp->setOperands(diffOp.getArguments());
  });

  return diffOp;
}

class DiffDeclPass : public DiffDeclPassBase<DiffDeclPass> {
  void runOnOperation() override {
    // process each funcOp
    getOperation()->walk([](func::FuncOp funcOp) {
      // if it is already a diff function, do nothing
      if (isDiffFunction(funcOp)) {
        return;
      }
      declareDiffFunction(funcOp);
    });
  }
};

std::unique_ptr<Pass> createDiffDeclPass() {
  return std::make_unique<DiffDeclPass>();
}

}  // namespace mlir::autodiff
