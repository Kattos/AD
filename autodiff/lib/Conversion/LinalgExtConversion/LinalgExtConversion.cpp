#include "Conversion/LinalgExtConversion/LinalgExtConversion.hpp"

#include "Util/Utils.hpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace autodiff {

class AllocRewriter : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern<bufferization::AllocTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp alloc,
                                PatternRewriter& rewriter) const override {
    auto type = alloc.getResult().getType();
    auto shape = type.getShape();
    auto elem = type.getElementType();

    rewriter.replaceOpWithNewOp<linalgext::InitTensorOp>(alloc, shape, elem);
    return success();
  }
};

class InitRewriter : public OpRewritePattern<linalgext::InitTensorOp> {
  using OpRewritePattern<linalgext::InitTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgext::InitTensorOp init,
                                PatternRewriter& rewriter) const override {
    auto tensor = init.getTensor();
    auto alloc = util::bufferization::alloc(tensor, rewriter);

    rewriter.replaceOp(init, alloc);
    return success();
  }
};

class AllocTensorToInitTensor
    : public impl::AllocTensorToInitTensorBase<AllocTensorToInitTensor> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AllocRewriter>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

class InitTensorToAllocTensor
    : public impl::InitTensorToAllocTensorBase<InitTensorToAllocTensor> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InitRewriter>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createAllocTensorToInitTensor() {
  return std::make_unique<AllocTensorToInitTensor>();
}

std::unique_ptr<Pass> createInitTensorToAllocTensor() {
  return std::make_unique<InitTensorToAllocTensor>();
}

}  // namespace autodiff
}  // namespace mlir
