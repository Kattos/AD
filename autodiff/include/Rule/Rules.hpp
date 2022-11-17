#ifndef RULES_HPP
#define RULES_HPP

#include "Utils.hpp"
// #include "mlir/Dialect/Tosa/IR/TosaOps.h"

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

    derivative.emplace_back(getInputDerivative(builder, op));
    return derivative;
  }

  Value getInputDerivative(OpBuilder& builder, OpTy op);

 private:
  SmallVector<Value, 1> derivative;
};

template <typename OpTy>
class BinaryOpRule : public OpRule<OpTy> {
 public:
  ValueRange concreteGetDerivatives(OpTy op) override {
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    derivatives.clear();
    derivatives.emplace_back(getLhsDerivative(builder, op));
    derivatives.emplace_back(getRhsDerivative(builder, op));

    return derivatives;
  }

  Value getLhsDerivative(OpBuilder& builder, OpTy op);
  Value getRhsDerivative(OpBuilder& builder, OpTy op);

 private:
  SmallVector<Value, 2> derivatives;
};

inline SmallVector<Value> gradients;
inline Operation* currentOp = nullptr;

// TODO: support different operand types/shapes
template <typename RuleTy>
ValueRange getGradients(Operation* op, Value grad) {
  if (op == currentOp && !gradients.empty()) {
    return gradients;
  }

  currentOp = op;

  auto rule = std::make_unique<RuleTy>();
  auto derivatives = rule->getDerivatives(op);
  assert(op->getNumOperands() == derivatives.size() && "Wrong operation rule");

  OpBuilder builder(op);
  builder.setInsertionPointAfterValue(grad);

  gradients.clear();
  for (size_t i = 0; i < derivatives.size(); ++i) {
    auto gradient = product(builder, grad, derivatives[i]);
    gradients.emplace_back(gradient);
  }
  return gradients;
}

template <typename RuleTy>
Value getGradient(Operation* op, Value grad, Value input) {
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    if (input == op->getOperand(i)) {
      if (op != currentOp || gradients.empty()) {
        getGradients<RuleTy>(op, grad);
      }
      return gradients[i];
    }
  }
  return nullptr;
}

// factory methods
ValueRange getGradients(Operation* op, Value grad);
Value getGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff

#endif  // RULES_HPP
