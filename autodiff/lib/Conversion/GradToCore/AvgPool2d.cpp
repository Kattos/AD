#include "Conversion/GradToCore/GradToCore.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

// TODO: implement integer version
// TODO: remove magic nums
Value dAvgPool2d(PatternRewriter& rewriter, Value output) {
  auto avg = output.getDefiningOp<grad::AvgPool2dOp>();

  if (!avg) {
    return nullptr;
  }

  auto x = avg.getX();
  auto dout = avg.getDout();
  auto dx = zeros(rewriter, x);

  auto kernelAttr = avg.getKernel();
  SmallVector<int64_t, 2> kernel;
  for (auto val : kernelAttr) {
    kernel.emplace_back(val.cast<IntegerAttr>().getInt());
  }

  auto strideAttr = avg.getStride();
  SmallVector<int64_t, 2> stride;
  for (auto val : strideAttr) {
    stride.emplace_back(val.cast<IntegerAttr>().getInt());
  }

  // build padded tensor
  auto padAttr = avg.getPad();
  SmallVector<int64_t, 8> pad;

  for (auto val : padAttr) {
    pad.emplace_back(val.cast<IntegerAttr>().getInt());
  }

  for (__attribute__((unused)) auto i : llvm::seq(0, 2)) {
    pad.insert(pad.begin(), 0);
    pad.emplace_back(0);
  }

  auto xType = x.getType().cast<RankedTensorType>();
  auto xElemType = xType.getElementType();
  auto xShape = xType.getShape();
  SmallVector<int64_t, 4> paddedShape;

  SmallVector<OpFoldResult, 4> lowIndices, highIndices;
  for (auto i : llvm::seq(0, 4)) {
    auto low = pad[i * 2];
    auto high = pad[i * 2 + 1];
    lowIndices.emplace_back(rewriter.getIndexAttr(low));
    highIndices.emplace_back(rewriter.getIndexAttr(high));
    paddedShape.emplace_back(xShape[i] + low + high);
  }

  auto zeroAttr = rewriter.getZeroAttr(xElemType);
  auto zero =
      createOp<arith::ConstantOp>(rewriter, xElemType, zeroAttr).getResult();

  auto paddedType = RankedTensorType::get(paddedShape, xElemType);
  auto paddedDx = createOp<tensor::PadOp>(rewriter, paddedType, dx, lowIndices,
                                          highIndices, zero)
                      .getResult();

  /*

  for i in range(dout.shape[0]):
    for j in range(dout.shape[1]):
      for k in range(dout.shape[2]):
        for l in range(dout.shape[3]):
          for m in range(kernel[0]):
            for n in range(kernel[1]):
              p = j * stride[0] + m
              q = k * stride[1] + n
              dx[i][p][q][l] += dout[i][j][k][l] / (kernel[0] * kernel[1])

  */

  // affine_map<(i, j, k, l, m, n) -> (i, j, k, l, m, n)>
  auto map = AffineMap::getMultiDimIdentityMap(6, rewriter.getContext());
  // affine_map<(i, j, k, l, m, n) -> (i, j, k, l)>
  auto mapForDout = map.getMajorSubMap(4);
  // affine_map<(i, j, k, l, m, n) -> (m, n)>
  auto mapForWindow = map.getMinorSubMap(2);

  SmallVector<AffineExpr, 6> exprs;

  for (auto i : llvm::seq(0, 6)) {
    exprs.emplace_back(rewriter.getAffineDimExpr(i));
  }

  // p = j * stride[0] + m
  exprs[1] = exprs[1] * stride[0] + exprs[4];
  // q = k * stride[1] + n
  exprs[2] = exprs[2] * stride[1] + exprs[5];
  exprs.pop_back_n(2);
  auto mapForDx = AffineMap::get(6, 0, exprs, rewriter.getContext());

  auto emptyWindow = createOp<tensor::EmptyOp>(rewriter, kernel, xElemType);
  auto window = zeros(rewriter, emptyWindow);

  auto resultTensorTypes = paddedDx.getType();
  auto inputs = ValueRange{dout, window};
  auto outputs = paddedDx;
  auto indexMaps = {mapForDout, mapForWindow, mapForDx};

  // reduction for innermost loop, parallel for others
  SmallVector<StringRef, 6> iteratorTypes;
  for (__attribute__((unused)) auto i : llvm::seq(0, 5)) {
    iteratorTypes.emplace_back(getParallelIteratorTypeName());
  }
  iteratorTypes.emplace_back(getReductionIteratorTypeName());

  // calculate kernel size
  auto size = 1.0;
  for (auto dim : kernel) {
    size *= dim;
  }
  auto sizeAttr = rewriter.getFloatAttr(getElementTypeOrSelf(x), size);
  auto sizeCst = createOp<arith::ConstantOp>(rewriter, sizeAttr);

  // dx[i][p][q][l] += dout[i][j][k][l] / (m * n)
  auto calculator = [&](OpBuilder& builder, Location loc, ValueRange args) {
    auto div = builder.create<arith::DivFOp>(loc, args[0], sizeCst);
    auto add = builder.create<arith::AddFOp>(loc, args[2], div);
    builder.create<linalg::YieldOp>(loc, add.getResult());
  };

  auto generic =
      createOp<linalg::GenericOp>(rewriter, resultTensorTypes, inputs, outputs,
                                  indexMaps, iteratorTypes, calculator);

  // extract dx from dpaddedx
  auto dPaddedX = generic->getResult(0);
  auto extOffsets = lowIndices;

  SmallVector<OpFoldResult, 4> extSizes, extStrides;
  for (auto i : llvm::seq(0, 4)) {
    extSizes.emplace_back(rewriter.getIndexAttr(xShape[i]));
    extStrides.emplace_back(rewriter.getIndexAttr(1));
  }

  auto extract = createOp<tensor::ExtractSliceOp>(
      rewriter, xType, dPaddedX, extOffsets, extSizes, extStrides);

  return extract;
}

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
