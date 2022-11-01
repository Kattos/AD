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

class SimpleADRewriter : public OpRewritePattern<ad::BackOp> {
  using OpRewritePattern<ad::BackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::BackOp back,
                                PatternRewriter& rewriter) const override {
    Operation* last = nullptr;
    for (auto operand : back->getOperands()) {
      last = rewriter.create<ad::OneslikeOp>(rewriter.getUnknownLoc(), operand);
    }
    rewriter.replaceOp(back, last->getResults());
    return success();
  }
};

class DiffPass : public DiffPassBase<DiffPass> {
  const StringRef PREFIX = "diff_";
  const StringRef VISIBILITY = "private";
  const StringRef IS_DIFF = "is_diff";

  // FIXME: what's the type
  auto getReturnOp(func::FuncOp func) {
    return func.getBody().rbegin()->rbegin();
  }

  auto getLastOp(func::FuncOp func) {
    return ++func.getBody().rbegin()->rbegin();
  }

  func::FuncOp genDiffFunction(func::FuncOp func) {
    // auto builder = OpBuilder::atBlockEnd(&func->getParentRegion()->back());
    OpBuilder builder(func->getContext());
    builder.setInsertionPointToEnd(&func->getParentRegion()->back());

    // copy primal op
    auto diff = dyn_cast<func::FuncOp>(builder.insert(func->clone()));
    auto diffName = PREFIX + func.getSymName();
    auto diffType = builder.getFunctionType(func.getFunctionType().getInputs(),
                                            func.getFunctionType().getInputs());
    auto visibility = builder.getStringAttr(VISIBILITY);

    diff.setSymName(diffName.str());
    diff.setFunctionType(diffType);
    diff.setSymVisibilityAttr(visibility);

    auto lastOp = getLastOp(diff);
    builder.setInsertionPointAfter(&*lastOp);

    auto returnOp = getReturnOp(diff);
    builder.create<ad::BackOp>(builder.getUnknownLoc(),
                               returnOp->getOperands().getType(),
                               returnOp->getOperands());
    returnOp->setOperands(diff.getArguments());

    // mark as differentiated
    diff->setAttr(IS_DIFF, builder.getBoolAttr(true));

    return diff;
  };

  void reverseTraverse(Value value) {
    value.dump();

    // if value is not function operands, do reverse BFS
    auto result = dyn_cast_or_null<OpResult>(value);
    if (nullptr == result) {
      return;
    }

    auto owner = result.getOwner();
    for (auto operand : owner->getOperands()) {
      reverseTraverse(operand);
    }
  }

  void backprop(Value value, Value grad, OpBuilder& builder) {
    value.dump();

    // if value is not function operands, do reverse BFS
    auto result = dyn_cast_or_null<OpResult>(value);
    if (nullptr == result) {
      return;
    }

    auto owner = result.getOwner();
    for (auto operand : owner->getOperands()) {
      backprop(operand, grad, builder);
    }
  }

  func::FuncOp simpleAD(func::FuncOp diff) {
    // get outputs
    OpBuilder builder(diff->getContext());

    for (auto block = diff.getBody().rbegin(); block != diff.getBody().rend();
         ++block) {
      for (auto op = block->rbegin(); op != block->rend(); ++op) {
        if (!isa<ad::BackOp>(*op)) {
          continue;
        }

        builder.setInsertionPointAfter(&*op);

        // process ad::BackOp
        auto operands = op->getOperands();
        for (auto operand : operands) {
          // backprop(operand, operand, builder);
        }

        break;
      }
    }

    return diff;
  }

  void runOnOperation() override {
    // TODO: replace with pass maybe
    getOperation()->walk([this](func::FuncOp func) {
      // avoid infinite loop
      auto isDiffOp = func->getAttr(IS_DIFF);
      if (isDiffOp) {
        return;
      }

      // gen `diff_func` for each `func` not differentiated
      auto diff = genDiffFunction(func);
      simpleAD(diff);
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<SimpleADRewriter>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::outs() << "failed\n";
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ad::ADDialect>();
  }
};

std::unique_ptr<Pass> createDiffPass() { return std::make_unique<DiffPass>(); }

}  // namespace mlir::autodiff