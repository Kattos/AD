#include "Conversion/GradToCore/GradToCore.hpp"

#include "Abs.cpp"
#include "Clamp.cpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {

class GradToCore : public impl::GradToCoreBase<GradToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<AbsToCore, 
                 RsqrtToCore,
                 LogToCore,
                 ExpToCore,
                 TanhToCore,
                 ClampToCore>(&getContext());
    // clang-format on
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createGradToCore() {
  return std::make_unique<GradToCore>();
}

}  // namespace mlir::autodiff
