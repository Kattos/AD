#include "Conversion/GradToCore/GradToCore.hpp"
#include "Rule/Utils.hpp"
#include "Utils.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

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

  auto x = avg.x();
  auto dout = avg.dout();
  auto dx = zeros(rewriter, x);

  SmallVector<int64_t> kernel;
  attrToArray(avg.kernelAttr(), kernel);

  SmallVector<int64_t> stride;
  attrToArray(avg.strideAttr(), stride);

  auto xType = x.getType().cast<RankedTensorType>();
  auto xElemType = xType.getElementType();

  auto paddedDx = pad2DTensor(rewriter, dx, avg.padAttr());

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

  auto bufferType = RankedTensorType::get(kernel, xElemType);
  auto emptyWindow = rewriter.create<bufferization::AllocTensorOp>(
      rewriter.getUnknownLoc(), bufferType, SmallVector<Value, 0>());
  // auto emptyWindow = createOp<tensor::EmptyOp>(rewriter, kernel, xElemType);
  auto window = zeros(rewriter, emptyWindow);

  auto resultTensorTypes = paddedDx.getType();
  auto inputs = ValueRange{dout, window};
  auto outputs = paddedDx;
  auto indexMaps = {mapForDout, mapForWindow, mapForDx};

  // parallel for dim0 and dim5, reduction for others
  SmallVector<StringRef, 6> iteratorTypes;
  for (__attribute__((unused)) auto i : llvm::seq(0, 4)) {
    iteratorTypes.push_back(getReductionIteratorTypeName());
  }
  iteratorTypes.insert(iteratorTypes.begin(), getParallelIteratorTypeName());
  iteratorTypes.emplace_back(getParallelIteratorTypeName());

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
  return unpad2DTensor(rewriter, generic->getResult(0), avg.padAttr());
}

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
