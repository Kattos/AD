#include "Conversion/ADToCore/ADToCore.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::autodiff {

Value constantInt(OpBuilder& builder, Type type, long value) {
  auto loc = builder.getUnknownLoc();
  auto attr = builder.getIntegerAttr(type, value);
  return builder.create<arith::ConstantOp>(loc, attr);
}

Value constantFloat(OpBuilder& builder, Type type, double value) {
  auto loc = builder.getUnknownLoc();
  auto attr = builder.getFloatAttr(type, value);
  return builder.create<arith::ConstantOp>(loc, attr);
}

Value constantFloatTensor(OpBuilder& builder, Type tensorType, double value) {
  auto loc = builder.getUnknownLoc();
  auto elemType = tensorType.cast<TensorType>().getElementType();
  auto constant = constantFloat(builder, elemType, value);
  auto valueType = RankedTensorType::get({1}, elemType);
  return builder.create<tensor::FromElementsOp>(loc, valueType, constant);
}

bool replaceWithValue(Operation* op, Type type, PatternRewriter& rewriter,
                      float value) {
  if (isa<TensorType>(type)) {
    rewriter.replaceOp(op, constantFloatTensor(rewriter, type, value));
  } else if (isa<FloatType>(type)) {
    rewriter.replaceOp(op, constantFloat(rewriter, type, value));
  } else if (isa<IntegerType>(type)) {
    rewriter.replaceOp(op, constantInt(rewriter, type, value));
  } else {
    return false;
  }
  return true;
}

class OneslikeToCore : public OpRewritePattern<ad::OneslikeOp> {
  using OpRewritePattern<ad::OneslikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::OneslikeOp oneslike,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(oneslike);
    return replaceWithValue(oneslike, oneslike.getType(), rewriter, 1.0)
               ? success()
               : failure();
  };
};

class ZeroslikeToCore : public OpRewritePattern<ad::ZeroslikeOp> {
  using OpRewritePattern<ad::ZeroslikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ZeroslikeOp zeroslike,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(zeroslike);
    return replaceWithValue(zeroslike, zeroslike.getType(), rewriter, 0.0)
               ? success()
               : failure();
  };
};

class ADToCore : public impl::ADToCoreBase<ADToCore> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<OneslikeToCore, ZeroslikeToCore>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createADToCore() { return std::make_unique<ADToCore>(); }

}  // namespace mlir::autodiff