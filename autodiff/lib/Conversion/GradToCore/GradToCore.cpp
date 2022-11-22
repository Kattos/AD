#include "Conversion/GradToCore/GradToCore.hpp"

#include "Abs.cpp"
#include "Clamp.cpp"
#include "GradToCoreImpl.cpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {

class GradToCore : public impl::GradToCoreBase<GradToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<AbsToCore, 
                 ClampToCore>(&getContext());
    // clang-format on
    grad::to_core::populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createGradToCore() {
  return std::make_unique<GradToCore>();
}

}  // namespace autodiff
}  // namespace mlir
