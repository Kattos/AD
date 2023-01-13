#include "Conversion/NablaLowering/NablaLowering.hpp"

#include "Util/Utils.hpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {

class NablaGradientLowering : public OpRewritePattern<nabla::GradientOp> {
  using OpRewritePattern<nabla::GradientOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nabla::GradientOp gradient,
                                PatternRewriter& rewriter) const override {
    auto target = gradient.getTarget();
    auto source = gradient.getSource();

    auto tape = util::tape::record(source, target, rewriter);
    rewriter.replaceOp(gradient, tape.get(source, rewriter));

    return success();
  }
};

class NablaLowering : public impl::NablaLoweringBase<NablaLowering> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<NablaGradientLowering>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createNablaLowering() {
  return std::make_unique<NablaLowering>();
}

}  // namespace autodiff
}  // namespace mlir