#include "Dialect/Grad/IR/Grad.hpp"

#define GET_OP_CLASSES
#include "Dialect/Grad/IR/Grad.cpp.inc"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace OpTrait {
namespace impl {

bool isTypeSame(Type a, Type b) {
  if (isa<IntegerType>(a))
    return isa<IntegerType>(b);

  else if (isa<FloatType>(a))
    return isa<FloatType>(b);

  // shaped type
  auto aShapedType = a.cast<ShapedType>();
  auto bShapedType = b.cast<ShapedType>();

  auto aElemType = aShapedType.getElementType();
  auto bElemType = bShapedType.getElementType();

  auto aShape = aShapedType.getShape();
  auto bShape = bShapedType.getShape();

  return aElemType == bElemType && aShape == bShape;
}

void prettyNotSameInputAndDerivativeType(Operation* op, Value input,
                                         Value derivative) {
  op->emitOpError() << "Types of `" << input << "` and `" << derivative
                    << "` are not the same";
}

LogicalResult verifySameInputAndDerivativeType(Operation* op) {
  auto inputs = op->getNumOperands();
  auto outputs = op->getNumResults();

  if (2 == inputs && 1 == outputs) {  // grad unary op
    auto x = op->getOperand(0);
    auto dx = op->getResult(0);

    if (!isTypeSame(x.getType(), dx.getType())) {
      prettyNotSameInputAndDerivativeType(op, x, dx);
      return failure();
    }

    return success();
  } else if (3 == inputs && 2 == outputs) {  // grad binary op
    auto lhs = op->getOperand(0);
    auto dlhs = op->getResult(0);

    auto rhs = op->getOperand(1);
    auto drhs = op->getResult(1);

    if (!isTypeSame(lhs.getType(), dlhs.getType())) {
      prettyNotSameInputAndDerivativeType(op, lhs, dlhs);
      return failure();
    }

    if (!isTypeSame(rhs.getType(), drhs.getType())) {
      prettyNotSameInputAndDerivativeType(op, rhs, drhs);
      return failure();
    }

    return success();
  }

  op->emitOpError("Operation is not from grad dialect");
  return failure();
}

}  // namespace impl
}  // namespace OpTrait

namespace autodiff {
namespace grad {

LogicalResult NablaOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  if (auto func = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
          *this, getFuncAttr())) {
    return success();
  }
  return emitOpError() << getFunc() << " not found";
}

}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
