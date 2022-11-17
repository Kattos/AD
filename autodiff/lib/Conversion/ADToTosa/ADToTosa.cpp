#include "Conversion/ADToTosa/ADToTosa.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {

template <typename OpTy>
class NumslikeConverter : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    float num = getNum();

    rewriter.setInsertionPointAfter(op);

    // auto loc = op->getLoc();
    auto input = op.getInput();
    auto type = input.getType().template dyn_cast<RankedTensorType>();

    assert(type && "input of `NumslikeOp` is not a ranked tensor\n");

    auto attr = DenseElementsAttr::get(type, {num});
    auto nums =
        rewriter.create<tosa::ConstOp>(rewriter.getUnknownLoc(), type, attr);

    rewriter.replaceOp(op, nums.getResult());

    return success();
  }

 protected:
  virtual float getNum() const = 0;
};

class OneslikeConverter : public NumslikeConverter<ad::OneslikeOp> {
  using NumslikeConverter<ad::OneslikeOp>::NumslikeConverter;

  float getNum() const override { return 1.0f; }
};

class ZeroslikeConverter : public NumslikeConverter<ad::ZeroslikeOp> {
  using NumslikeConverter<ad::ZeroslikeOp>::NumslikeConverter;

  float getNum() const override { return 0.0f; }
};

class ADToTosa : public impl::ADToTosaBase<ADToTosa> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<OneslikeConverter, ZeroslikeConverter>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createADToTosa() { return std::make_unique<ADToTosa>(); }

}  // namespace mlir::autodiff
