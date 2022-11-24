#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"

#include "Rule/Utils.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace concrete {

Value toClamp(PatternRewriter& rewriter, Value unary) {
  auto abstract = unary.getDefiningOp<grad::AbstractUnaryOp>();

  if (!abstract) {
    return nullptr;
  }

  auto resultTypes = abstract->getResultTypes();
  auto operands = abstract->getOperands();
  abstract->removeAttr("op");
  auto attrs = abstract->getAttrs();

  return createOp<grad::ClampOp>(rewriter, resultTypes, operands, attrs);
}

}  // namespace concrete
}  // namespace grad

class GradAbstractToConcrete
    : public impl::GradAbstractToConcreteBase<GradAbstractToConcrete> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    grad::concrete::populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createGradAbstractToConcrete() {
  return std::make_unique<GradAbstractToConcrete>();
}

}  // namespace autodiff
}  // namespace mlir
