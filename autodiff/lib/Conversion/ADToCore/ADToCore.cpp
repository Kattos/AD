#include "Conversion/ADToCore/ADToCore.hpp"

#include "Abs.cpp"
#include "GradAbs.cpp"
#include "Numslike.cpp"
#include "Placeholder.cpp"
#include "Rsqrt.cpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {
class ADToCore : public impl::ADToCoreBase<ADToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<OneslikeToCore, 
                 ZeroslikeToCore, 
                 PlaceholderToCore,
                 AbsToCore,
                 RsqrtToCore,
                 GradAbsToCore>(&getContext());
    // clang-format on
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createADToCore() { return std::make_unique<ADToCore>(); }

}  // namespace mlir::autodiff
