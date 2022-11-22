#include "Conversion/ADToCore/ADToCore.hpp"
#include "Rule/Utils.hpp"

namespace mlir {
namespace autodiff {

LogicalResult verifySameShape(Value a, Value b) {
  auto aType = a.getType();
  auto bType = b.getType();

  if (!isa<ShapedType>(aType) || !isa<ShapedType>(bType)) {
    return failure();
  }

  auto aShape = aType.cast<ShapedType>().getShape();
  auto bShape = bType.cast<ShapedType>().getShape();
  return success(aShape == bShape);
}

class BroadcastToCore : public OpRewritePattern<ad::BroadcastOp> {
  using OpRewritePattern<ad::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::BroadcastOp broadcast,
                                PatternRewriter& rewriter) const override {
    auto from = broadcast.getFrom();
    auto to = broadcast.getTo();

    if (succeeded(verifySameShape(from, to))) {
      rewriter.replaceOp(broadcast, from);
      return success();
    }

    auto zero = zeros(rewriter, to);
    auto reshape = createOp<tosa::AddOp>(rewriter, to.getType(), from, zero);
    rewriter.replaceOp(broadcast, reshape.getResult());
    return success();
  }
};

class ReduceToCore : public OpRewritePattern<ad::ReduceOp> {
  using OpRewritePattern<ad::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ReduceOp reduce,
                                PatternRewriter& rewriter) const override {
    auto from = reduce.getFrom();
    auto to = reduce.getTo();

    if (succeeded(verifySameShape(from, to))) {
      rewriter.replaceOp(reduce, from);
      return success();
    }

    auto elemType = from.getType().getElementType();
    auto fromShape = from.getType().getShape();
    auto toShape = to.getType().getShape();

    auto fromVec = fromShape.vec();
    auto toVec = toShape.vec();

    while (fromVec.size() > toVec.size()) {
      toVec.emplace_back(1);
    }

    for (size_t i = 0; i < fromVec.size(); ++i) {
      if (fromVec[i] == toVec[i]) {
        continue;
      }

      if (toVec[i] != 1) {
        reduce->emitOpError() << fromShape << " cannot reduce to " << toShape;
        return failure();
      }

      fromVec[i] = 1;
      auto type = RankedTensorType::get(fromVec, elemType);
      auto axis = rewriter.getI64IntegerAttr(i);

      from = createOp<tosa::ReduceSumOp>(rewriter, type, from, axis);
    }

    auto attr = rewriter.getI64ArrayAttr(toShape);
    auto reshape =
        createOp<tosa::ReshapeOp>(rewriter, to.getType(), from, attr);

    rewriter.replaceOp(reduce, reshape.getResult());
    return success();
  }
};

}  // namespace autodiff
}  // namespace mlir
