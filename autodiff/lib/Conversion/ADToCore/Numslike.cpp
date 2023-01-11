#include "Conversion/ADToCore/ADToCore.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

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
  return builder.create<tensor::FromElementsOp>(loc, tensorType, constant);
}

bool replaceWithValue(Operation* op, Type type, PatternRewriter& rewriter,
                      float value) {
  if (type.isa<TensorType>()) {
    rewriter.replaceOp(op, constantFloatTensor(rewriter, type, value));
  } else if (type.isa<FloatType>()) {
    rewriter.replaceOp(op, constantFloat(rewriter, type, value));
  } else if (type.isa<IntegerType>()) {
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
    return success(
        replaceWithValue(oneslike, oneslike.getType(), rewriter, 1.0));
  };
};

class ZeroslikeToCore : public OpRewritePattern<ad::ZeroslikeOp> {
  using OpRewritePattern<ad::ZeroslikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ZeroslikeOp zeroslike,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(zeroslike);
    return success(
        replaceWithValue(zeroslike, zeroslike.getType(), rewriter, 0.0));
  };
};

}  // namespace mlir::autodiff
