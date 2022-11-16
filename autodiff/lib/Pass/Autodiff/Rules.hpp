#ifndef AUTODIFF_DERIVATIVES_HPP
#define AUTODIFF_DERIVATIVES_HPP

#include "ADUtils.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

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

    auto derivatives = std::make_unique<SmallVector<Value, 1>>();
    derivatives->push_back(getInputDerivative(builder, op));
    return *derivatives;
  }

  Value getInputDerivative(OpBuilder& builder, OpTy op);
};

template <typename OpTy>
class BinaryOpRule : public OpRule<OpTy> {
 public:
  ValueRange concreteGetDerivatives(OpTy op) override {
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto derivatives = std::make_unique<SmallVector<Value, 2>>();
    derivatives->push_back(getLhsDerivative(builder, op));
    derivatives->push_back(getRhsDerivative(builder, op));

    return *derivatives;
  }

  Value getLhsDerivative(OpBuilder& builder, OpTy op);
  Value getRhsDerivative(OpBuilder& builder, OpTy op);
};

// TODO: support different operand types/shapes
// FIXME: not works well
template <typename RuleTy>
ValueRange getGradients(Operation* op, Value grad) {
  auto rule = std::make_unique<RuleTy>();
  auto derivatives = rule->getDerivatives(op);
  assert(op->getNumOperands() == derivatives.size() && "Wrong operation rule");

  OpBuilder builder(op);
  builder.setInsertionPointAfterValue(grad);

  SmallVector<Value> gradients;

  for (size_t i = 0; i < derivatives.size(); ++i) {
    auto gradient = product(builder, grad, derivatives[i]);
    gradients.push_back(gradient);
  }
  return gradients;
}

template <typename RuleTy>
Value getGradient(Operation* op, Value grad, Value input) {
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    if (input == op->getOperand(i)) {
      return getGradients<RuleTy>(op, grad)[i];
    }
  }
  return nullptr;
}

// factory methods
// TODO: remove this function if no use
ValueRange getGradients(Operation* op, Value grad);
Value getGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff

#endif  // AUTODIFF_DERIVATIVES_HPP
