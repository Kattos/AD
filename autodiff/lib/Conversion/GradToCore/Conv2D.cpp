#include "Conversion/GradToCore/GradToCore.hpp"
#include "Utils.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
            # out[i][j][k][l] += dx[i][p][q][l] * weight[i][m][n][l] + bias[l]
            p = j * stride[0] + m
            q = k * stride[1] + n
            dx[i][p][q][l] += dout[i][j][k][l] * weight[i][m][n][l]
            dweight[i][m][n][l] += dout[i][j][k][l] * x[i][p][q][l]
            dbias[l] += dout[i][j][k][l] / (m * n)

*/

Value dConv2DInput(PatternRewriter& rewriter, Value output) {
  auto conv = output.getDefiningOp<grad::Conv2DOp>();
  if (!conv) {
    return nullptr;
  }

  auto loc = rewriter.getUnknownLoc();
  auto ctx = rewriter.getContext();

  auto x = conv.x();
  auto dx = rewriter.create<ad::ZeroslikeOp>(loc, x).getResult();
  auto dout = conv.dout();
  auto weight = conv.weight();

  // get strides
  SmallVector<int64_t> stride;
  attrToArray(conv.strideAttr(), stride);

  // get dilations
  SmallVector<int64_t> dilations;
  // TODO: support dilation
  attrToArray(conv.dilationAttr(), dilations);

  // build padded tensor
  auto paddedDx = pad2DTensor(rewriter, dx, conv.padAttr());

  // build window
  auto weightType = weight.getType().cast<ShapedType>();
  auto weightShape = weightType.getShape();
  auto elemType = weightType.getElementType();

  auto windowShape = {weightShape[1], weightShape[2]};
  auto bufferType = RankedTensorType::get(windowShape, elemType);
  // auto emptyWindow =
  //     rewriter.create<tensor::EmptyOp>(loc, windowShape, elemType);
  auto emptyWindow = rewriter.create<bufferization::AllocTensorOp>(
      rewriter.getUnknownLoc(), bufferType, SmallVector<Value, 0>());
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

  return unpad2DTensor(rewriter, generic->getResult(0), conv.padAttr());
}

/*

for i in range(dout.shape[0]):
  for j in range(dout.shape[1]):
    for k in range(dout.shape[2]):
      for l in range(dout.shape[3]):
        bias[l] += dout[i][j][k][l]

*/

Value dConv2DBias(PatternRewriter& rewriter, Value output) {
  auto conv = output.getDefiningOp<grad::Conv2DOp>();
  if (!conv) {
    return nullptr;
  }

  auto loc = rewriter.getUnknownLoc();
  auto bias = conv.bias();
  auto dbias = rewriter.create<ad::ZeroslikeOp>(loc, bias).getResult();
  auto dout = conv.dout();

  constexpr auto DIM_COUNT = 4;
  auto mapForDout = rewriter.getMultiDimIdentityMap(DIM_COUNT);
  auto mapForBias = mapForDout.getMinorSubMap(1);

  auto resultTensorTypes = dbias.getType();
  auto inputs = dout;
  auto outputs = dbias;
  auto indexMaps = {mapForDout, mapForBias};

  SmallVector<StringRef, DIM_COUNT> iteratorTypes;
  for (auto i = 0; i < DIM_COUNT - 1; i++) {
    iteratorTypes.emplace_back(getReductionIteratorTypeName());
  }
  iteratorTypes.emplace_back(getParallelIteratorTypeName());

  auto calculator = [](OpBuilder& builder, Location loc, ValueRange args) {
    auto dout = args[0];
    auto bias = args[1];
    auto add = builder.create<arith::AddFOp>(loc, dout, bias);
    builder.create<linalg::YieldOp>(loc, add.getResult());
  };

  auto generic = rewriter.create<linalg::GenericOp>(loc, resultTensorTypes,
                                                    inputs, outputs, indexMaps,
                                                    iteratorTypes, calculator);

  return generic->getResult(0);
}

class GradConv2DToCore : public OpRewritePattern<grad::Conv2DOp> {
  using OpRewritePattern<grad::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::Conv2DOp conv,
                                PatternRewriter& rewriter) const override {
    auto loc = rewriter.getUnknownLoc();
    auto ctx = rewriter.getContext();

    auto dout = conv.dout();
    auto x = conv.x();
    auto dx = rewriter.create<ad::ZeroslikeOp>(loc, x).getResult();
    auto weight = conv.weight();
    auto dweight = rewriter.create<ad::ZeroslikeOp>(loc, weight).getResult();
    auto bias = conv.bias();
    auto dbias = rewriter.create<ad::ZeroslikeOp>(loc, bias).getResult();

    // get strides
    SmallVector<int64_t> stride;
    attrToArray(conv.strideAttr(), stride);

    // get dilations
    SmallVector<int64_t> dilations;
    attrToArray(conv.dilationAttr(), dilations);

    // build padded tensor
    auto padX = pad2DTensor(rewriter, x, conv.padAttr());
    auto padDx = pad2DTensor(rewriter, dx, conv.padAttr());

    // build affine maps
    constexpr auto DIM_COUNT = 6;
    constexpr auto SYM_COUNT = 0;

    SmallVector<AffineExpr> exprs;

    auto getAffineExprs =
        [&](std::initializer_list<int> dims) -> SmallVector<AffineExpr> {
      if (!exprs.empty()) {
        exprs.clear();
      }
      exprs.reserve(dims.size());
      for (auto dim : dims) {
        exprs.emplace_back(rewriter.getAffineDimExpr(dim));
      }
      return exprs;
    };

    auto getAffineMap = [&](std::initializer_list<int> dims) -> AffineMap {
      return AffineMap::get(DIM_COUNT, SYM_COUNT, getAffineExprs(dims), ctx);
    };

    // affine_map<(i, j, k, l, m, n) -> (i, j, k, l)>
    auto doutMap = getAffineMap({0, 1, 2, 3});
    // affine_map<(i, j, k, l, m, n) -> (i, m, n, l)>
    auto weightMap = getAffineMap({0, 4, 5, 3});
    // affine_map<(i, j, k, l, m, n) -> (l)>
    auto biasMap = getAffineMap({3});

    // affine_map<(i, j, k, l, m, n) -> (i, j * s[0] + m, k * s[1] + n, l)>
    auto xExprs = getAffineExprs({0, 1, 2, 3});
    xExprs[1] = xExprs[1] * stride[0] + rewriter.getAffineDimExpr(4);
    xExprs[2] = xExprs[2] * stride[1] + rewriter.getAffineDimExpr(5);
    auto xMap = AffineMap::get(DIM_COUNT, SYM_COUNT, xExprs, ctx);

    auto resTypes = TypeRange{padX.getType(), weight.getType(), bias.getType()};
    auto ins = ValueRange{dout, padX, weight, bias};
    auto outs = ValueRange{padDx, dweight, dbias};
    auto idxMaps = ArrayRef<AffineMap>(
        {doutMap, xMap, weightMap, biasMap, xMap, weightMap, biasMap});

    SmallVector<StringRef, DIM_COUNT> iterTypes;
    for (__attribute__((unused)) auto i : llvm::seq(0, DIM_COUNT - 2)) {
      iterTypes.push_back(getReductionIteratorTypeName());
    }
    iterTypes.insert(iterTypes.begin(), getParallelIteratorTypeName());
    iterTypes.emplace_back(getParallelIteratorTypeName());

    auto weightH = weight.getType().cast<TensorType>().getDimSize(1);
    auto weightW = weight.getType().cast<TensorType>().getDimSize(2);
    auto sizeValue = static_cast<double>(weightH * weightW);
    auto sizeAttr = rewriter.getF32FloatAttr(sizeValue);
    auto size = rewriter.create<arith::ConstantOp>(loc, sizeAttr).getResult();

    auto calculator = [size](OpBuilder& builder, Location loc,
                             ValueRange args) {
      auto dout = args[0];
      auto x = args[1];
      auto weight = args[2];
      // auto bias = args[3];
      auto dx = args[4];
      auto dweight = args[5];
      auto dbias = args[6];

      // dx
      auto dxProd = builder.create<arith::MulFOp>(loc, dout, weight);
      auto dxSum = builder.create<arith::AddFOp>(loc, dx, dxProd);

      // dweight
      auto dweightProd = builder.create<arith::MulFOp>(loc, dout, x);
      auto dweightSum =
          builder.create<arith::AddFOp>(loc, dweight, dweightProd);

      // dbias
      auto dbiasQuo = builder.create<arith::DivFOp>(loc, dout, size);
      auto dbiasSum = builder.create<arith::AddFOp>(loc, dbias, dbiasQuo);

      builder.create<linalg::YieldOp>(
          loc, ValueRange{dxSum.getResult(), dweightSum.getResult(),
                          dbiasSum.getResult()});
    };

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resTypes, ins, outs, idxMaps, iterTypes, calculator);

    auto retX = unpad2DTensor(rewriter, generic->getResult(0), conv.padAttr());
    auto retWeight = generic->getResult(1);
    auto retBias = generic->getResult(2);

    rewriter.replaceOp(conv, {retX, retWeight, retBias});

    return success();
  }
};

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
