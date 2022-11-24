#include "Conversion/GradToCore/GradToCore.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
