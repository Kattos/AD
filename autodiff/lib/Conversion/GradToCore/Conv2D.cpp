#include "Conversion/GradToCore/GradToCore.hpp"
#include "Utils.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

/*

for i in range(dout.shape[0]):
  for j in range(dout.shape[1]):
    for k in range(dout.shape[2]):
      for l in range(dout.shape[3]):
        for m in range(weight.shape[1]):
          for n in range(weight.shape[2]): # window = weight.shape[1:3]
            p = j * stride[0] + m
            q = k * stride[1] + n
            dx[i][p][q][l] += dout[i][j][k][l] * weight[i][m][n][l]

*/

Value dConv2DInput(PatternRewriter& rewriter, Value output) {
  auto conv = output.getDefiningOp<grad::Conv2DOp>();
  if (!conv) {
    return nullptr;
  }

  auto loc = rewriter.getUnknownLoc();
  auto ctx = rewriter.getContext();

  auto x = conv.getX();
  auto dx = rewriter.create<ad::ZeroslikeOp>(loc, x).getResult();
  auto dout = conv.getDout();
  auto weight = conv.getWeight();

  // get strides
  SmallVector<int64_t> stride;
  attrToArray(conv.getStrideAttr(), stride);

  // get dilations
  SmallVector<int64_t> dilations;
  // TODO: support dilation
  attrToArray(conv.getDilationAttr(), dilations);

  // build padded tensor
  auto paddedDx = pad2DTensor(rewriter, dx, conv.getPadAttr());

  // build window
  auto weightType = weight.getType().cast<ShapedType>();
  auto weightShape = weightType.getShape();
  auto elemType = weightType.getElementType();

  auto windowShape = {weightShape[1], weightShape[2]};
  auto emptyWindow =
      rewriter.create<tensor::EmptyOp>(loc, windowShape, elemType);
  auto window = rewriter.create<ad::ZeroslikeOp>(loc, emptyWindow);

  // build affine maps
  // total nested loop level
  constexpr auto DIM_COUNT = 6;
  constexpr auto SYM_COUNT = 0;
  // input tensor rank
  constexpr auto RANKS = 4;

  // affine_map<(i, j, k, l, m, n) -> (i, j, k, l, m, n)>
  auto identityMap = rewriter.getMultiDimIdentityMap(DIM_COUNT);
  // affine_map<(i, j, k, l, m, n) -> (i, j, k, l)>
  auto mapForDout = identityMap.getMajorSubMap(RANKS);
  // affine_map<(i, j, k, l, m, n) -> (m, n)>
  auto mapForWindow = identityMap.getMinorSubMap(DIM_COUNT - RANKS);

  // affine_map<(i, j, k, l, m, n) -> (i, m, n, l)>
  SmallVector<AffineExpr, RANKS> weightExprs;
  for (auto i : {0, 4, 5, 3}) {
    weightExprs.emplace_back(rewriter.getAffineDimExpr(i));
  }
  auto mapForWeight = AffineMap::get(DIM_COUNT, SYM_COUNT, weightExprs, ctx);

  // affine_map<(i, j, k, l, m, n) -> (i, j * s[0] + m, k * s[1] + n, l)>
  SmallVector<AffineExpr, DIM_COUNT> dxExprs;
  for (auto i : llvm::seq(0, DIM_COUNT)) {
    dxExprs.emplace_back(rewriter.getAffineDimExpr(i));
  }
  dxExprs[1] = dxExprs[1] * stride[0] + dxExprs[4];
  dxExprs[2] = dxExprs[2] * stride[1] + dxExprs[5];
  dxExprs.resize(RANKS);
  auto mapForDx = AffineMap::get(DIM_COUNT, SYM_COUNT, dxExprs, ctx);

  // build generic op
  auto resultTensorTypes = paddedDx.getType().cast<TensorType>();
  auto inputs = ValueRange{dout, window, weight};
  auto outputs = paddedDx;
  auto indexMaps = {mapForDout, mapForWindow, mapForWeight, mapForDx};

  // parallel for dim0 and dim5, reduction for others
  SmallVector<StringRef, DIM_COUNT> iteratorTypes;
  for (__attribute__((unused)) auto i : llvm::seq(0, DIM_COUNT - 2)) {
    iteratorTypes.push_back(getReductionIteratorTypeName());
  }
  iteratorTypes.insert(iteratorTypes.begin(), getParallelIteratorTypeName());
  iteratorTypes.emplace_back(getParallelIteratorTypeName());

  // dx[i][p][q][l] += dout[i][j][k][l] * weight[i][m][n][l]
  auto calculator = [&](OpBuilder& builder, Location loc, ValueRange args) {
    auto mul = builder.create<arith::MulFOp>(loc, args[0], args[2]);
    auto add = builder.create<arith::AddFOp>(loc, args[3], mul);
    builder.create<linalg::YieldOp>(loc, add.getResult());
  };

  auto generic = rewriter.create<linalg::GenericOp>(loc, resultTensorTypes,
                                                    inputs, outputs, indexMaps,
                                                    iteratorTypes, calculator);

  return unpad2DTensor(rewriter, generic->getResult(0), conv.getPadAttr());
}

Value dConv2DBias(PatternRewriter& rewriter, Value output) {
  auto conv = output.getDefiningOp<grad::Conv2DOp>();
  if (!conv) {
    return nullptr;
  }

  auto loc = rewriter.getUnknownLoc();
  auto bias = conv.getBias();

  return rewriter.create<ad::OneslikeOp>(loc, bias);
}

class GradConv2DToCore : public OpRewritePattern<grad::Conv2DOp> {
  using OpRewritePattern<grad::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::Conv2DOp conv,
                                PatternRewriter& rewriter) const override {
    return success();
  }
};

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
