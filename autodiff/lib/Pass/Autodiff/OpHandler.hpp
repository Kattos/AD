#ifndef AUTODIFF_OPHANDLER_HPP
#define AUTODIFF_OPHANDLER_HPP

#include "ADUtils.hpp"
#include "Dialect/AD/IR/AD.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::autodiff {

// template <typename OpTy, typename... Args>
// OpTy createOp(OpBuilder& builder, Args&&... args) {
//   return builder.create<OpTy>(builder.getUnknownLoc(),
//                               std::forward<Args>(args)...);
// }

class OpHandlerInterface {
 public:
  virtual ~OpHandlerInterface() = default;

  virtual SmallVector<Value>* handleOp(Operation* op, Value grad,
                                       OpBuilder& builder) = 0;
};

template <typename OpTy>
class OpHandler : public OpHandlerInterface {
 public:
  virtual ~OpHandler() = default;

  SmallVector<Value>* handleOp(Operation* op, Value grad,
                               OpBuilder& builder) override {
    if (!isa<OpTy>(*op)) {
      return nullptr;
    }

    auto contribution = handleOp(op->getOperands(), grad, builder);
    if (!contribution) {
      exit(1 && "Unsupported operation detected\n");
    }

    return contribution;
  }

  virtual SmallVector<Value>* handleOp(OperandRange operands, Value grad,
                                       OpBuilder& builder) = 0;
};

template <typename OpTy>
class UnaryOpHandler : public OpHandler<OpTy> {
 public:
  virtual ~UnaryOpHandler() = default;

  SmallVector<Value>* handleOp(OperandRange operands, Value grad,
                               OpBuilder& builder) override {
    SmallVector<Value>* contributions = new SmallVector<Value>();
    this->input = operands[0];
    this->grad = grad;

    auto inputGrad = getGrad(builder);
    contributions->push_back(inputGrad);
    return contributions;
  }

  virtual Value getGrad(OpBuilder& builder) = 0;

 protected:
  Value input;
  Value grad;
};

template <typename OpTy>
class BinaryOpHandler : public OpHandler<OpTy> {
 public:
  virtual ~BinaryOpHandler() = default;

  SmallVector<Value>* handleOp(OperandRange operands, Value grad,
                               OpBuilder& builder) override {
    SmallVector<Value>* contributions = new SmallVector<Value>();
    this->lhs = operands[0];
    this->rhs = operands[1];
    this->grad = grad;

    auto lhsGrad = getLhsGrad(builder);
    auto rhsGrad = getRhsGrad(builder);
    contributions->push_back(lhsGrad);
    contributions->push_back(rhsGrad);
    return contributions;
  }

  virtual Value getLhsGrad(OpBuilder& builder) = 0;
  virtual Value getRhsGrad(OpBuilder& builder) = 0;

 protected:
  Value lhs;
  Value rhs;
  Value grad;
};

//===----------------------------------------------------------------------===//
// AddOpHandler
//===----------------------------------------------------------------------===//
class AddOpHandler : public BinaryOpHandler<tosa::AddOp> {
 public:
  Value getLhsGrad(OpBuilder& builder) override {
    return oneslike(lhs, builder);
  }

  Value getRhsGrad(OpBuilder& builder) override {
    return oneslike(rhs, builder);
  }

  static OpHandler& instance() {
    static std::unique_ptr<OpHandler> instance(new AddOpHandler());
    return *instance;
  }

 private:
  Value oneslike(Value input, OpBuilder& builder) {
    return createOp<ad::OneslikeOp>(builder, input);
  }
};

//===----------------------------------------------------------------------===//
// SubOpHandler
//===----------------------------------------------------------------------===//
class SubOpHandler : public BinaryOpHandler<tosa::SubOp> {
 public:
  Value getLhsGrad(OpBuilder& builder) override {
    return oneslike(lhs, builder);
  }

  Value getRhsGrad(OpBuilder& builder) override {
    auto ones = oneslike(rhs, builder);
    return builder.create<tosa::NegateOp>(builder.getUnknownLoc(),
                                          ones.getType(), ones);
  }

  static OpHandler& instance() {
    static std::unique_ptr<OpHandler> instance(new SubOpHandler());
    return *instance;
  }

 private:
  Value oneslike(Value input, OpBuilder& builder) {
    return createOp<ad::OneslikeOp>(builder, input);
  }
};

//===----------------------------------------------------------------------===//
// MulOpHandler
//===----------------------------------------------------------------------===//
class MulOpHandler : public BinaryOpHandler<tosa::MulOp> {
 public:
  Value getLhsGrad(OpBuilder& builder) override {
    auto type = lhs.getType();
    auto shift = noShift(builder);
    return createOp<tosa::MulOp>(builder, type, rhs, grad, shift);
  }

  Value getRhsGrad(OpBuilder& builder) override {
    auto type = lhs.getType();
    auto shift = noShift(builder);
    return createOp<tosa::MulOp>(builder, type, lhs, grad, shift);
  }

  static OpHandler& instance() {
    static std::unique_ptr<OpHandler> instance(new MulOpHandler());
    return *instance;
  }

 private:
  IntegerAttr noShift(OpBuilder& builder) {
    return builder.getI32IntegerAttr(0);
  }
};

//===----------------------------------------------------------------------===//
// LogOpHandler
//===----------------------------------------------------------------------===//
class LogOpHandler : public UnaryOpHandler<tosa::LogOp> {
 public:
  Value getGrad(OpBuilder& builder) override {
    return createOp<tosa::ReciprocalOp>(builder, input.getType(), input);
  }

  static OpHandler& instance() {
    static std::unique_ptr<OpHandler> instance(new LogOpHandler());
    return *instance;
  }
};

//===----------------------------------------------------------------------===//
// HandlerFactory
//===----------------------------------------------------------------------===//
class HandlerFactory {
 public:
  static OpHandlerInterface& getOpHandler(Operation* op) {
    // TODO: support more operations
    if (isa<tosa::AddOp>(*op)) {
      return AddOpHandler::instance();
    } else if (isa<tosa::SubOp>(*op)) {
      return SubOpHandler::instance();
    } else if (isa<tosa::MulOp>(*op)) {
      return MulOpHandler::instance();
    } else if (isa<tosa::LogOp>(*op)) {
      // return LogOpHandler::instance().handleOp(op, grad, builder);
      return LogOpHandler::instance();
    } else {
      exit(1 && "Unsupported operation detected");
    }
  }

  static SmallVector<Value>* getResults(Operation* op, Value grad,
                                        OpBuilder& builder) {
    auto& handler = getOpHandler(op);
    return handler.handleOp(op, grad, builder);
  }

  static Value getContribution(Operation* op, Value grad, Value value,
                               OpBuilder& builder) {
    auto grads = getResults(op, grad, builder);
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      if (value == op->getOperand(i)) {
        return (*grads)[i];
      }
    }
    return nullptr;
  }
};

}  // namespace mlir::autodiff

#endif  // AUTODIFF_OPHANDLER_HPP
