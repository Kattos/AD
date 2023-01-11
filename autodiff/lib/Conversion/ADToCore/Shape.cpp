#include "Conversion/ADToCore/ADToCore.hpp"
#include "Rule/Utils.hpp"

namespace mlir {
namespace autodiff {

LogicalResult verifySameShape(Value a, Value b) {
  auto aType = a.getType();
  auto bType = b.getType();

  if (!aType.isa<ShapedType>() || !bType.isa<ShapedType>()) {
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
    auto from = broadcast.from();
    auto to = broadcast.to();

    if (succeeded(verifySameShape(from, to))) {
      rewriter.replaceOp(broadcast, from);
      return success();
    }

    auto zero = zeros(rewriter, to);
    auto reshape = createOp<tosa::AddOp>(rewriter, to.getType(), from, zero);
    rewriter.replaceOp(broadcast, reshape.output());
    return success();
  }
};

class ReduceToCore : public OpRewritePattern<ad::ReduceOp> {
  using OpRewritePattern<ad::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ReduceOp reduce,
                                PatternRewriter& rewriter) const override {
    auto from = reduce.from();
    auto to = reduce.to();

    if (succeeded(verifySameShape(from, to))) {
      rewriter.replaceOp(reduce, from);
      return success();
    }

    auto elemType = from.getType().cast<ShapedType>().getElementType();
    auto fromShape = from.getType().cast<ShapedType>().getShape();
    auto toShape = to.getType().cast<ShapedType>().getShape();

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

    rewriter.replaceOp(reduce, reshape.output());
    return success();
  }
};

}  // namespace autodiff
}  // namespace mlir
