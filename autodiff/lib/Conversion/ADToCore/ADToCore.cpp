#include "Conversion/ADToCore/ADToCore.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {

Value constantTensor(OpBuilder& builder, float value) {
  auto loc = builder.getUnknownLoc();
  auto attr = builder.getF32FloatAttr(value);
  auto constant = builder.create<arith::ConstantOp>(loc, attr).getResult();
  auto type = RankedTensorType::get({1}, builder.getF32Type());

  return builder.create<tensor::FromElementsOp>(loc, type, constant);
}

class OneslikeToCore : public OpRewritePattern<ad::OneslikeOp> {
  using OpRewritePattern<ad::OneslikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::OneslikeOp oneslike,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(oneslike);
    rewriter.replaceOp(oneslike, constantTensor(rewriter, 1.0));
    return success();
  };
};

class ZeroslikeToCore : public OpRewritePattern<ad::ZeroslikeOp> {
  using OpRewritePattern<ad::ZeroslikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ZeroslikeOp zeroslike,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(zeroslike);
    rewriter.replaceOp(zeroslike, constantTensor(rewriter, 0.0));
    return success();
  };
};

class ADToCore : public impl::ADToCoreBase<ADToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<OneslikeToCore>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createADToCore() { return std::make_unique<ADToCore>(); }

}  // namespace mlir::autodiff