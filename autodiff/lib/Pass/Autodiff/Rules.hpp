#ifndef AUTODIFF_DERIVATIVES_HPP
#define AUTODIFF_DERIVATIVES_HPP

#include "ADUtils.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::autodiff {

class OpRuleInterface {
 public:
  virtual ~OpRuleInterface() = default;

  // return df/dinput for each input
  virtual ValueRange getDerivatives(Operation* op) = 0;
};

template <typename OpTy>
class OpRule : public OpRuleInterface {
 public:
  ValueRange getDerivatives(Operation* op) override {
    auto ty = dyn_cast<OpTy>(op);
    if (!ty) {
      return {};
    }

    return concreteGetDerivatives(ty);
  }

  virtual ValueRange concreteGetDerivatives(OpTy op) = 0;
};

template <typename OpTy>
class UnaryOpRule : public OpRule<OpTy> {
 public:
  ValueRange concreteGetDerivatives(OpTy op) override {
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    SmallVector<Value, 1>* derivatives = new SmallVector<Value, 1>();
    derivatives->push_back(getInputDerivative(builder, op));
    return *derivatives;
  }

  Value getInputDerivative(OpBuilder& builder, OpTy op);
};

using LogOpRule = UnaryOpRule<tosa::LogOp>;
using ExpOpRule = UnaryOpRule<tosa::ExpOp>;

template <typename OpTy>
class BinaryOpRule : public OpRule<OpTy> {
 public:
  ValueRange concreteGetDerivatives(OpTy op) override {
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    SmallVector<Value, 2>* derivatives = new SmallVector<Value, 2>();
    derivatives->push_back(getLhsDerivative(builder, op));
    derivatives->push_back(getRhsDerivative(builder, op));
    return *derivatives;
  }

  Value getLhsDerivative(OpBuilder& builder, OpTy op);
  Value getRhsDerivative(OpBuilder& builder, OpTy op);
};

using AddOpRule = BinaryOpRule<tosa::AddOp>;
using SubOpRule = BinaryOpRule<tosa::SubOp>;
using MulOpRule = BinaryOpRule<tosa::MulOp>;

template <typename RuleTy>
ValueRange getGradients(Operation* op, Value grad) {
  static auto rule = std::unique_ptr<RuleTy>(new RuleTy());
  auto derivatives = rule->getDerivatives(op);

  OpBuilder builder(op);
  auto func = dyn_cast<func::FuncOp>(op->getParentOp());
  assert(func && "Unsupported nested func body");

  builder.setInsertionPointAfterValue(grad);

  SmallVector<Value> gradients;
  for (auto d : derivatives) {
    auto g = product(builder, grad, d);
    gradients.push_back(g);
  }

  return gradients;
}

template <typename RuleTy>
Value getGradient(Operation* op, Value grad, Value input) {
  auto gradients = getGradients<RuleTy>(op, grad);

  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    if (input == op->getOperand(i)) {
      return getGradients<RuleTy>(op, grad)[i];
    }
  }

  return nullptr;
}

// factory methods
ValueRange getGradients(Operation* op, Value grad);
Value getGradient(Operation* op, Value grad, Value input);

inline ValueRange getExpGradients(Operation* op, Value grad) {
  return getGradients<ExpOpRule>(op, grad);
}

}  // namespace mlir::autodiff

#endif  // AUTODIFF_DERIVATIVES_HPP
