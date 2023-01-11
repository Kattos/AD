#include "MathRules.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::autodiff {

using CmpFn = function_ref<Value(OpBuilder& builder, Value lhs, Value rhs)>;
using NegFn = function_ref<Value(OpBuilder& builder, Value input)>;

// TODO: support tensor type
Value gradAbs(OpBuilder& builder, Operation* op, CmpFn eq, CmpFn gt, NegFn ne) {
  auto input = op->getOperand(0);
  auto zero = zeros(builder, input);
  auto type = input.getType();

  auto eqZero = eq(builder, input, zero);
  auto isEq = createOp<scf::IfOp>(builder, type, eqZero, true);

  // begin [input == 0]
  auto eqBuilder = isEq.getThenBodyBuilder();
  createOp<scf::YieldOp>(eqBuilder, zero);

  // begin [input != 0]
  auto neqBuilder = isEq.getElseBodyBuilder();
  auto one = ones(neqBuilder, input);

  auto gtZero = gt(neqBuilder, input, zero);
  auto isGt = createOp<scf::IfOp>(neqBuilder, type, gtZero, true);

  // begin [input > 0]
  auto gtBuilder = isGt.getThenBodyBuilder();
  createOp<scf::YieldOp>(gtBuilder, one);
  // end [input > 0]

  // begin [input < 0]
  auto ltBuilder = isGt.getElseBodyBuilder();
  auto neg = ne(ltBuilder, one);
  createOp<scf::YieldOp>(ltBuilder, neg);
  // end [input < 0]

  createOp<scf::YieldOp>(neqBuilder, isGt->getResult(0));
  // end [input != 0]

  return isEq->getResult(0);
  // end [input == 0]
}

template <>
Value MathAbsRule::getInputDerivative(OpBuilder& builder, math::AbsOp absf) {
  CmpFn eq = [](OpBuilder& builder, Value lhs, Value rhs) -> Value {
    return createOp<arith::CmpFOp>(builder, arith::CmpFPredicate::OEQ, lhs,
                                   rhs);
  };

  CmpFn gt = [](OpBuilder& builder, Value lhs, Value rhs) -> Value {
    return createOp<arith::CmpFOp>(builder, arith::CmpFPredicate::OGT, lhs,
                                   rhs);
  };

  NegFn ne = [](OpBuilder& builder, Value input) -> Value {
    return createOp<arith::NegFOp>(builder, input);
  };

  return gradAbs(builder, absf, eq, gt, ne);
}

}  // namespace mlir::autodiff
