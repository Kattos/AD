#include "ArithRules.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::autodiff {

// TODO: support tensor type
Value gradMin(OpBuilder& builder, Operation* op, Value cond) {
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
Value ArithMinFRule::getLhsDerivative(OpBuilder& builder, arith::MinFOp minf) {
  auto le = cmpF(builder, minf, arith::CmpFPredicate::OLE);
  return gradMin(builder, minf, le);
}

template <>
Value ArithMinFRule::getRhsDerivative(OpBuilder& builder, arith::MinFOp minf) {
  auto gt = cmpF(builder, minf, arith::CmpFPredicate::OGT);
  return gradMin(builder, minf, gt);
}

template <>
Value ArithMinSIRule::getLhsDerivative(OpBuilder& builder,
                                       arith::MinSIOp minsi) {
  auto le = cmpI(builder, minsi, arith::CmpIPredicate::sle);
  return gradMin(builder, minsi, le);
}

template <>
Value ArithMinSIRule::getRhsDerivative(OpBuilder& builder,
                                       arith::MinSIOp minsi) {
  auto gt = cmpI(builder, minsi, arith::CmpIPredicate::sgt);
  return gradMin(builder, minsi, gt);
}

template <>
Value ArithMinUIRule::getLhsDerivative(OpBuilder& builder,
                                       arith::MinUIOp minui) {
  auto le = cmpI(builder, minui, arith::CmpIPredicate::ule);
  return gradMin(builder, minui, le);
}

template <>
Value ArithMinUIRule::getRhsDerivative(OpBuilder& builder,
                                       arith::MinUIOp minui) {
  auto gt = cmpI(builder, minui, arith::CmpIPredicate::ugt);
  return gradMin(builder, minui, gt);
}

}  // namespace mlir::autodiff
