#include "Conversion/ADToCore/ADToCore.hpp"

#include "Numslike.cpp"
#include "Shape.cpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {
namespace ad {

Value buildScalarTensor(PatternRewriter& rewriter, Value input, Value output) {
  auto op = output.getDefiningOp<ad::ScalarTensorOp>();
  if (!op) {
    return output;
  }

  return createOp<tensor::FromElementsOp>(rewriter, output.getType(), input);
}

}  // namespace ad

class ADToCore : public impl::ADToCoreBase<ADToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<OneslikeToCore, 
                 ZeroslikeToCore, 
                 BroadcastToCore,
                 ReduceToCore>(&getContext());
    ad::populateWithGenerated(patterns);
    // clang-format on
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createADToCore() { return std::make_unique<ADToCore>(); }
}  // namespace autodiff
}  // namespace mlir
