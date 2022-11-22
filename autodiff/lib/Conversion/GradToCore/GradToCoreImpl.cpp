#include "Conversion/GradToCore/GradToCore.hpp"
#include "Rule/Utils.hpp"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace to_core {

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

}  // namespace to_core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
