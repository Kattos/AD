#include "Dialect/AD/IR/AD.hpp"
#include "Pass/Simplify/Passes.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {

class MultiplyByOne : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp mul,
                                PatternRewriter& rewriter) const override {
    auto lhs = mul.getInput1();
    auto rhs = mul.getInput2();

    if (lhs.getDefiningOp<ad::OneslikeOp>()) {
      rewriter.replaceOp(mul, rhs);
    } else if (rhs.getDefiningOp<ad::OneslikeOp>()) {
      rewriter.replaceOp(mul, lhs);
    }

    return success();
  }
};

class MultiplyByZero : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp mul,
                                PatternRewriter& rewriter) const override {
    auto lhs = mul.getInput1();
    auto rhs = mul.getInput2();

    if (lhs.getDefiningOp<ad::OneslikeOp>()) {
      rewriter.replaceOp(mul, lhs);
    } else if (rhs.getDefiningOp<ad::OneslikeOp>()) {
      rewriter.replaceOp(mul, rhs);
    }

    return success();
  }
};

class PlusByZero : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp add,
                                PatternRewriter& rewriter) const override {
    auto lhs = add.getInput1();
    auto rhs = add.getInput2();

    if (lhs.getDefiningOp<ad::ZeroslikeOp>()) {
      rewriter.replaceOp(add, rhs);
    } else if (rhs.getDefiningOp<ad::ZeroslikeOp>()) {
      rewriter.replaceOp(add, lhs);
    }

    return success();
  }
};

class SimplifyPass : public SimplifyPassBase<SimplifyPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MultiplyByOne>(&getContext());

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createADSimplifyPass() {
  return std::make_unique<SimplifyPass>();
}

}  // namespace mlir::autodiff
