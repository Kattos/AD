#include "Conversion/GradToCore/GradToCore.hpp"
#include "Rule/Utils.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

Value add(PatternRewriter& rewriter, Value lhs, Value rhs) {
  return createOp<tosa::AddOp>(rewriter, lhs.getType(), lhs, rhs);
}

Value mul(PatternRewriter& rewriter, Value lhs, Value rhs) {
  auto shift = rewriter.getI32IntegerAttr(0);
  return createOp<tosa::MulOp>(rewriter, lhs.getType(), lhs, rhs, shift);
}

Value negate(PatternRewriter& rewriter, Value tensor) {
  return createOp<tosa::NegateOp>(rewriter, tensor.getType(), tensor);
}

Value exp(PatternRewriter& rewriter, Value tensor) {
  return createOp<tosa::ExpOp>(rewriter, tensor.getType(), tensor);
}

Value reciprocal(PatternRewriter& rewriter, Value tensor) {
  return createOp<tosa::ReciprocalOp>(rewriter, tensor.getType(), tensor);
}

Value oneslike(PatternRewriter& rewriter, Value tensor) {
  return createOp<ad::OneslikeOp>(rewriter, tensor);
}

Value broadcast(PatternRewriter& rewriter, Value from, Value to) {
  return createOp<ad::BroadcastOp>(rewriter, from, to);
}

Value reduce(PatternRewriter& rewriter, Value from, Value to) {
  return createOp<ad::ReduceOp>(rewriter, from, to);
}

Value drsqrt(PatternRewriter& rewriter, Value tensor) {
  auto type = getElementTypeOrSelf(tensor);

  auto coefficientAttr = rewriter.getFloatAttr(type, -0.5);
  auto coefficient = createOp<arith::ConstantOp>(rewriter, coefficientAttr);
  auto coefficientTensor = createOp<ad::ToTensorOp>(rewriter, coefficient);

  auto exponentAttr = rewriter.getFloatAttr(type, -1.5);
  auto exponent = createOp<arith::ConstantOp>(rewriter, exponentAttr);
  auto exponentTensor = createOp<ad::ToTensorOp>(rewriter, exponent);

  auto pow =
      createOp<tosa::PowOp>(rewriter, tensor.getType(), tensor, exponentTensor);

  return product(rewriter, pow, coefficientTensor);
}

Value dabs(PatternRewriter& rewriter, Value tensor) {
  auto type = tensor.getType().dyn_cast<TensorType>();
  if (!type) {
    return nullptr;
  }

  auto condType = RankedTensorType::get(type.getShape(), rewriter.getI1Type());

  auto zero = zeros(rewriter, tensor);
  auto gtCond = createOp<tosa::GreaterOp>(rewriter, condType, tensor, zero);
  auto ltCond = createOp<tosa::GreaterOp>(rewriter, condType, zero, tensor);

  auto ge = createOp<tosa::CastOp>(rewriter, type, gtCond);
  auto le = createOp<tosa::CastOp>(rewriter, type, ltCond);

  auto negate = createOp<tosa::NegateOp>(rewriter, type, le);
  return sum(rewriter, ge, negate);
}

Value dGreaterEqual(PatternRewriter& rewriter, Value first, Value second) {
  auto type = first.getType().dyn_cast<TensorType>();
  if (!type) {
    return nullptr;
  }

  auto elemType = type.getElementType();
  auto condType = RankedTensorType::get(type.getShape(), rewriter.getI1Type());

  auto gtCond = createOp<tosa::GreaterOp>(rewriter, condType, first, second);
  auto eqCond = createOp<tosa::EqualOp>(rewriter, condType, first, second);

  auto gt = createOp<tosa::CastOp>(rewriter, type, gtCond);
  auto eq = createOp<tosa::CastOp>(rewriter, type, eqCond);

  auto half = createOp<ad::ScalarTensorOp>(rewriter, elemType, 0.5);
  return sum(rewriter, gt, product(rewriter, eq, half));
}

template <typename AttrTy>
Value clampHelper(PatternRewriter& rewriter, Value tensor, Attribute min,
                  Attribute max) {
  auto minAttr = min.dyn_cast<AttrTy>();
  auto maxAttr = max.dyn_cast<AttrTy>();

  if (!minAttr || !maxAttr) {
    return nullptr;
  }

  auto one = ones(rewriter, tensor);
  auto type = tensor.getType();
  auto shape = type.cast<ShapedType>().getShape();

  auto cmpTensor = [&](AttrTy attr) -> Value {
    auto value = createOp<arith::ConstantOp>(rewriter, attr);
    auto scalar = createOp<ad::ScalarTensorOp>(rewriter, value);
    auto cmpTensor = product(rewriter, one, scalar);
    return cmpTensor;
  };

  auto minTensor = cmpTensor(minAttr);
  auto maxTensor = cmpTensor(maxAttr);

  auto condType = RankedTensorType::get(shape, rewriter.getI1Type());
  auto ge =
      createOp<tosa::GreaterEqualOp>(rewriter, condType, tensor, minTensor);
  auto le =
      createOp<tosa::GreaterEqualOp>(rewriter, condType, maxTensor, tensor);

  return createOp<tosa::LogicalAndOp>(rewriter, condType, ge, le);
}

Value intClampHelper(PatternRewriter& rewriter, Value tensor, Attribute min,
                     Attribute max) {
  return clampHelper<IntegerAttr>(rewriter, tensor, min, max);
}

Value floatClampHelper(PatternRewriter& rewriter, Value tensor, Attribute min,
                       Attribute max) {
  return clampHelper<FloatAttr>(rewriter, tensor, min, max);
}

// TODO: implement integer version
Value avgPool2dHelper(PatternRewriter& rewriter, Value output) {
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

  for (auto i = 0; i < 2; i++) {
    pad.insert(pad.begin(), 0);
    pad.emplace_back(0);
  }

  auto xType = x.getType();
  auto xElemType = xType.getElementType();
  auto xShape = xType.getShape();
  SmallVector<int64_t, 4> paddedShape;

  SmallVector<OpFoldResult, 4> lowIndices, highIndices;
  for (auto i = 0; i < 4; i++) {
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
              dx[i][p][q][l] = dout[i][j][k][l] / (kernel[0] * kernel[1])

  */

  // affine_map<(i, j, k, l, m, n) -> (i, j, k, l, m, n)>
  auto map = AffineMap::getMultiDimIdentityMap(6, rewriter.getContext());
  // affine_map<(i, j, k, l, m, n) -> (i, j, k, l)>
  auto mapForDout = map.getMajorSubMap(4);
  // affine_map<(i, j, k, l, m, n) -> (m, n)>
  auto mapForWindow = map.getMinorSubMap(2);

  SmallVector<AffineExpr, 6> exprs;

  for (auto i = 0; i < 6; i++) {
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
  for (auto i = 0; i < 5; i++) {
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

  // dx[i][p][q][l] = dout[i][j][k][l] / (m * n)
  auto calculator = [&](OpBuilder& builder, Location loc, ValueRange args) {
    auto div = createOp<arith::DivFOp>(builder, args[0], sizeCst);
    auto add = createOp<arith::AddFOp>(builder, args[2], div);
    createOp<linalg::YieldOp>(builder, add.getResult());
  };

  auto generic =
      createOp<linalg::GenericOp>(rewriter, resultTensorTypes, inputs, outputs,
                                  indexMaps, iteratorTypes, calculator);

  // extract dx from dpaddedx
  auto dPaddedX = generic->getResult(0);
  auto extOffsets = lowIndices;

  SmallVector<OpFoldResult, 4> extSizes, extStrides;
  for (auto i = 0; i < 4; i++) {
    extSizes.emplace_back(rewriter.getIndexAttr(xShape[i]));
    extStrides.emplace_back(rewriter.getIndexAttr(1));
  }

  auto extract = createOp<tensor::ExtractSliceOp>(
      rewriter, xType.cast<RankedTensorType>(), dPaddedX, extOffsets, extSizes,
      extStrides);

  return extract;
}

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
