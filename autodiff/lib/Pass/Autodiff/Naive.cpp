#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "OpHandler.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {
class ToPattern : public OpRewritePattern<ad::ToOp> {
  using OpRewritePattern<ad::ToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ToOp to,
                                PatternRewriter& rewriter) const override {
    rewriter.eraseOp(to);
    return success();
  }
};

class FromPattern : public OpRewritePattern<ad::FromOp> {
  using OpRewritePattern<ad::FromOp>::OpRewritePattern;

  using TAPE = std::unordered_map<Operation*, Value>;

  void backprop(Value input, Value grad, PatternRewriter& rewriter) const {
    auto loc = rewriter.getUnknownLoc();
    if (isa<BlockArgument>(input)) {
      rewriter.create<ad::ReturnOp>(loc, input, grad);
      return;
    }

    auto op = input.getDefiningOp();

    // TODO: find right grads
    auto curGrad = rewriter.create<ad::ZeroslikeOp>(loc, input);
    auto newGrad =
        rewriter.create<tosa::AddOp>(loc, curGrad.getType(), curGrad, grad);

    auto operands = op->getOperands();
    SmallVector<Value>* contributions =
        HandlerFactory::getResults(op, newGrad, rewriter);

    // update the gradients of previous operations
    for (size_t i = 0; i < operands.size(); ++i) {
      backprop(operands[i], (*contributions)[i], rewriter);
    }
  }

  void backpropWithTape(Value input, Value grad, PatternRewriter& rewriter,
                        TAPE& tape) const {
    auto loc = rewriter.getUnknownLoc();
    if (isa<BlockArgument>(input)) {
      rewriter.create<ad::ReturnOp>(loc, input, grad);
      return;
    }
    auto op = input.getDefiningOp();
    if (isa<ad::ToOp>(*op)) {
      rewriter.create<ad::ReturnOp>(loc, input, grad);
      return;
    }

    auto curGrad = tape[op];
    if (!curGrad) {
      curGrad = rewriter.create<ad::ZeroslikeOp>(loc, input);
    }
    auto newGrad =
        rewriter.create<tosa::AddOp>(loc, grad.getType(), curGrad, grad);
    tape[op] = newGrad;

    auto operands = op->getOperands();
    SmallVector<Value>* contributions =
        HandlerFactory::getResults(op, newGrad, rewriter);

    // update the gradients of previous operations
    for (size_t i = 0; i < operands.size(); ++i) {
      backpropWithTape(operands[i], (*contributions)[i], rewriter, tape);
    }
  }

  LogicalResult matchAndRewrite(ad::FromOp from,
                                PatternRewriter& rewriter) const override {
    auto grad =
        rewriter
            .create<ad::OneslikeOp>(rewriter.getUnknownLoc(), from.getInput())
            .getOutput();

    TAPE tape;
    tape[from] = grad;

    backpropWithTape(from.getInput(), grad, rewriter, tape);

    rewriter.eraseOp(from);
    return success();
  }
};

class NaivePass : public NaivePassBase<NaivePass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());

    RewritePatternSet patterns(&getContext());
    patterns.add<ToPattern>(&getContext());
    patterns.add<FromPattern>(&getContext());

    getOperation()->walk([&](func::FuncOp func) {
      auto isDiff = func->getAttr("diff");
      if (!isDiff) {
        return;
      }

      // rewrite `ad.from` and `ad.to`
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        llvm::outs() << "failed\n";
        signalPassFailure();
      }

      std::map<unsigned int, Value> index2grad;

      func->walk([&](ad::ReturnOp ret) {
        auto index = ret.getArgument().cast<BlockArgument>().getArgNumber();
        index2grad[index] = ret.getGrad();

        ret->erase();
      });

      func->walk([&](func::ReturnOp ret) {
        for (size_t i = 0; i < ret->getOperands().size(); ++i) {
          auto index = ret->getOperand(i).cast<BlockArgument>().getArgNumber();
          auto grad = index2grad[index];
          ret->setOperand(i, grad);
        }
      });
    });
  }
};

std::unique_ptr<Pass> createADNaivePass() {
  return std::make_unique<NaivePass>();
}

}  // namespace mlir::autodiff