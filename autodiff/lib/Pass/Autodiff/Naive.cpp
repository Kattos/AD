#include <map>
#include <utility>

#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {

// erase `to` op
class ToPattern : public OpRewritePattern<ad::ToOp> {
  using OpRewritePattern<ad::ToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ToOp to,
                                PatternRewriter& rewriter) const override {
    rewriter.eraseOp(to);
    return success();
  }
};

class NaivePass : public NaivePassBase<NaivePass> {
  // void backprop(Operation* op, Value contribution, OpBuilder& builder) {
  //   auto zeros =
  //       builder.create<ad::ZeroslikeOp>(builder.getUnknownLoc(),
  //       contribution)
  //           .getOutput();
  //   auto grad = builder.create<tosa::AddOp>(
  //       builder.getUnknownLoc(), zeros.getType(), zeros, contribution);
  //   for (auto operand : op->getOperands()) {
  //     // TODO: support other operations
  //     if (isa<BlockArgument>(operand)) {
  //       builder.create<ad::ReturnOp>(builder.getUnknownLoc(), operand, grad);
  //     } else {
  //       // TODO: calculate contribution
  //       backprop(dyn_cast<OpResult>(operand).getOwner(), contribution,
  //       builder);
  //     }
  //   }
  // }

  void backprop(Value value, Value contribution, OpBuilder& builder) {
    if (isa<BlockArgument>(value)) {
      builder.create<ad::ReturnOp>(builder.getUnknownLoc(), value,
                                   contribution);
      return;
    }

    auto owner = dyn_cast<OpResult>(value).getOwner();

    // TODO: find current grads
    auto currentGrad =
        builder.create<ad::ZeroslikeOp>(builder.getUnknownLoc(), value);
    auto grad = builder.create<tosa::AddOp>(builder.getUnknownLoc(),
                                            currentGrad.getType(), currentGrad,
                                            contribution);

    // TODO: calculate contribution
    auto newContribution = grad;

    for (auto operand : owner->getOperands()) {
      backprop(operand, newContribution, builder);
    }
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](func::FuncOp func) {
      auto isDiff = func->getAttr("diff");
      if (!isDiff) {
        return;
      }

      func->walk([&](ad::FromOp from) {
        builder.setInsertionPointAfter(from);

        auto contribution = builder
                                .create<ad::OneslikeOp>(builder.getUnknownLoc(),
                                                        from.getInput())
                                .getOutput();

        // backprop(from, contribution, builder);
        backprop(from.getInput(), contribution, builder);
        from->erase();
      });

      std::map<unsigned int, Value> index2grad;

      func->walk([&](ad::ReturnOp ret) {
        auto index = dyn_cast<BlockArgument>(ret.getArgument()).getArgNumber();
        index2grad[index] = ret.getGrad();
        ret->erase();
      });

      func->walk([&](func::ReturnOp ret) {
        for (size_t i = 0; i < ret->getOperands().size(); ++i) {
          auto index =
              dyn_cast<BlockArgument>(ret->getOperand(i)).getArgNumber();
          auto grad = index2grad[index];
          ret->setOperand(i, grad);
        }
      });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<ToPattern>(&getContext());

    ConversionTarget target(getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      llvm::outs() << "failed\n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createADNaivePass() {
  return std::make_unique<NaivePass>();
}

}  // namespace mlir::autodiff