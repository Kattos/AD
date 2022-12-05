#include "Conversion/GradToCore/GradToCore.hpp"

#include "AvgPool2d.cpp"
#include "Conv2D.cpp"
#include "GradToCoreImpl.cpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {

class GradToCore : public impl::GradToCoreBase<GradToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    grad::core::populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createGradToCore() {
  return std::make_unique<GradToCore>();
}

}  // namespace autodiff
}  // namespace mlir
