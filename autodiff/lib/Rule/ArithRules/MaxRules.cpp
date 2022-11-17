#include "ArithRules.hpp"
#include "CmpUtils.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::autodiff {

// TODO: support tensor type
Value gradMax(OpBuilder& builder, Operation* op, Value cond) {
  auto lhs = op->getOperand(0);
  auto ifOp = createOp<scf::IfOp>(builder, lhs.getType(), cond, true);

  // begin [cond]
  auto condBuilder = ifOp.getThenBodyBuilder();
  createOp<scf::YieldOp>(condBuilder, ones(condBuilder, lhs));
  // end [cond]

  // begin [not cond]
  auto notCondBuilder = ifOp.getElseBodyBuilder();
  createOp<scf::YieldOp>(notCondBuilder, zeros(notCondBuilder, lhs));
  // end [not cond]

  return ifOp.getResult(0);
}

template <>
Value ArithMaxFRule::getLhsDerivative(OpBuilder& builder, arith::MaxFOp maxf) {
  auto ge = cmpF(builder, maxf, arith::CmpFPredicate::OGE);
  return gradMax(builder, maxf, ge);
}

template <>
Value ArithMaxFRule::getRhsDerivative(OpBuilder& builder, arith::MaxFOp maxf) {
  auto lt = cmpF(builder, maxf, arith::CmpFPredicate::OLT);
  return gradMax(builder, maxf, lt);
}

template <>
Value ArithMaxSIRule::getLhsDerivative(OpBuilder& builder,
                                       arith::MaxSIOp maxsi) {
  auto ge = cmpI(builder, maxsi, arith::CmpIPredicate::sge);
  return gradMax(builder, maxsi, ge);
}

template <>
Value ArithMaxSIRule::getRhsDerivative(OpBuilder& builder,
                                       arith::MaxSIOp maxsi) {
  auto lt = cmpI(builder, maxsi, arith::CmpIPredicate::slt);
  return gradMax(builder, maxsi, lt);
}

template <>
Value ArithMaxUIRule::getLhsDerivative(OpBuilder& builder,
                                       arith::MaxUIOp maxui) {
  auto ge = cmpI(builder, maxui, arith::CmpIPredicate::uge);
  return gradMax(builder, maxui, ge);
}

template <>
Value ArithMaxUIRule::getRhsDerivative(OpBuilder& builder,
                                       arith::MaxUIOp maxui) {
  auto lt = cmpI(builder, maxui, arith::CmpIPredicate::ult);
  return gradMax(builder, maxui, lt);
}

}  // namespace mlir::autodiff
