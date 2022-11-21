#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"

#include "AbstractBinary.cpp"
#include "AbstractUnary.cpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {

class GradAbstractToConcrete
    : public impl::GradAbstractToConcreteBase<GradAbstractToConcrete> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<AbstractUnaryToConcrete, AbstractBinaryToConcrete>(&getContext());
    // clang-format on
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createGradAbstractToConcrete() {
  return std::make_unique<GradAbstractToConcrete>();
}

}  // namespace autodiff
}  // namespace mlir
