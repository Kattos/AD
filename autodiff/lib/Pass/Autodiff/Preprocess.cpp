#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {
class PreprocessPass : public PreprocessPassBase<PreprocessPass> {
  const StringRef PREFIX = "diff_";
  const StringRef ISDIFF = "diff";

  // FIXME: what's the type
  auto getReturnOp(func::FuncOp func) { return func.rbegin()->rbegin(); }

  auto getLastOp(func::FuncOp func) { return ++func.rbegin()->rbegin(); }

  auto getFirstOp(func::FuncOp func) { return func.begin()->begin(); }

  func::FuncOp genDiffFunction(func::FuncOp func) {
    OpBuilder builder(func->getContext());
    builder.setInsertionPointToEnd(&func->getParentRegion()->back());

    // copy primal op
    auto diff = dyn_cast<func::FuncOp>(builder.insert(func->clone()));
    auto diffName = PREFIX + func.getSymName();
    auto diffType = builder.getFunctionType(func.getArgumentTypes(),
                                            func.getArgumentTypes());

    diff.setSymName(diffName.str());
    diff.setFunctionType(diffType);
    diff.setPrivate();

    // mark as differentiated
    diff->setAttr(ISDIFF, builder.getBoolAttr(true));

    // FIXME: seems like `to` op is useless
    // insert `to` for each input at the beginning of func body
    auto firstOp = getFirstOp(diff);
    builder.setInsertionPoint(&*firstOp);
    for (auto argument : diff.getArguments()) {
      // as placeholder
      auto to = builder.create<ad::ToOp>(builder.getUnknownLoc(), argument);

      // argument.replaceAllUsesExcept(to, to);
    }

    // insert `from` for each output at the end of func body
    auto lastOp = getLastOp(diff);
    builder.setInsertionPointAfter(&*lastOp);

    auto returnOp = getReturnOp(diff);
    for (auto operand : returnOp->getOperands()) {
      builder.create<ad::FromOp>(builder.getUnknownLoc(), operand);
    }

    returnOp->setOperands(diff.getArguments());

    return diff;
  };

  void runOnOperation() override {
    // TODO: replace with rewriter maybe
    getOperation()->walk([this](func::FuncOp func) {
      // avoid infinite loop
      auto isDiffOp = func->getAttr(ISDIFF);
      if (isDiffOp) {
        return;
      }

      // gen `diff_func` for each `func` not differentiated
      genDiffFunction(func);
    });
  }
};

std::unique_ptr<Pass> createADPreprocessPass() {
  return std::make_unique<PreprocessPass>();
}

}  // namespace mlir::autodiff