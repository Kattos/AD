#include "Utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

SmallVector<int64_t> attrToArray(ArrayAttr attrs, SmallVector<int64_t>& array) {
  array.reserve(attrs.size());
  for (auto attr : attrs) {
    array.emplace_back(attr.cast<IntegerAttr>().getInt());
  }
  return array;
}

Value pad2DTensor(PatternRewriter& rewriter, Value tensor, ArrayAttr padAttr) {
  SmallVector<int64_t> pad;
  attrToArray(padAttr, pad);

  // 4 dummy elements
  pad.reserve(pad.size() + 4);
  for (__attribute__((unused)) auto i : llvm::seq(0, 2)) {
    pad.insert(pad.begin(), 0);
    pad.emplace_back(0);
  }

  auto type = tensor.getType().cast<RankedTensorType>();
  auto elemType = type.getElementType();
  auto shape = type.getShape();

  SmallVector<int64_t> paddedShape(shape);
  SmallVector<OpFoldResult, 4> low, high;
  for (auto i : llvm::seq(0, 4)) {
    auto l = pad[2 * i];
    auto h = pad[2 * i + 1];
    low.emplace_back(rewriter.getIndexAttr(l));
    high.emplace_back(rewriter.getIndexAttr(h));
    paddedShape[i] += l + h;
  }

  auto loc = rewriter.getUnknownLoc();
  auto zeroAttr = rewriter.getZeroAttr(elemType);
  auto zero =
      rewriter.create<arith::ConstantOp>(loc, elemType, zeroAttr).getResult();

  auto paddedType = RankedTensorType::get(paddedShape, elemType);
  auto paddedTensor =
      rewriter.create<tensor::PadOp>(loc, paddedType, tensor, low, high, zero);

  return paddedTensor;
}

Value unpad2DTensor(PatternRewriter& rewriter, Value paddedTensor,
                    ArrayAttr padAttr) {
  SmallVector<int64_t> pad;
  attrToArray(padAttr, pad);

  // 4 dummy elements
  pad.reserve(pad.size() + 4);
  for (__attribute__((unused)) auto i : llvm::seq(0, 2)) {
    pad.insert(pad.begin(), 0);
    pad.emplace_back(0);
  }

  auto paddedType = paddedTensor.getType().cast<RankedTensorType>();
  auto elemType = paddedType.getElementType();
  auto paddedShape = paddedType.getShape();

  SmallVector<int64_t> shape(paddedShape);
  SmallVector<OpFoldResult, 4> offset, size, stride;

  for (auto i : llvm::seq(0, 4)) {
    auto l = pad[2 * i];
    auto h = pad[2 * i + 1];
    shape[i] -= (l + h);

    offset.emplace_back(rewriter.getIndexAttr(l));
    size.emplace_back(rewriter.getIndexAttr(shape[i]));
    stride.emplace_back(rewriter.getIndexAttr(1));
  }

  auto loc = rewriter.getUnknownLoc();
  auto type = RankedTensorType::get(shape, elemType);
  auto tensor = rewriter.create<tensor::ExtractSliceOp>(loc, type, paddedTensor,
                                                        offset, size, stride);

  return tensor;
}

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
