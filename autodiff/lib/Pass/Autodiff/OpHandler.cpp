#include "Dialect/AD/IR/AD.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::autodiff {

template <typename OpTy, typename... Args>
OpTy createOp(PatternRewriter& rewriter, Args&&... args) {
  return rewriter.create<OpTy>(rewriter.getUnknownLoc(),
                               std::forward<Args>(args)...);
}

template <typename OpTy>
class OpHandler {
 public:
  virtual ~OpHandler() = default;

  SmallVector<Value>* handleOp(Operation* op, Value grad,
                               PatternRewriter& rewriter) {
    if (!isa<OpTy>(*op)) {
      return nullptr;
    }

    auto contribution = handleOp(op->getOperands(), grad, rewriter);
    if (!contribution) {
      exit(1 && "Unsupported operation detected\n");
    }

    return contribution;
  }

  virtual SmallVector<Value>* handleOp(OperandRange operands, Value grad,
                                       PatternRewriter& rewriter) = 0;
};

template <typename OpTy>
class UnaryOpHandler : public OpHandler<OpTy> {
 public:
  virtual ~UnaryOpHandler() = default;

  SmallVector<Value>* handleOp(OperandRange operands, Value grad,
                               PatternRewriter& rewriter) override {
    SmallVector<Value>* contributions = new SmallVector<Value>();
    this->input = operands[0];
    this->grad = grad;

    auto inputGrad = getGrad(rewriter);
    contributions->push_back(inputGrad);
    return contributions;
  }

  virtual Value getGrad(PatternRewriter& rewriter) = 0;

 protected:
  Value input;
  Value grad;
};

template <typename OpTy>
class BinaryOpHandler : public OpHandler<OpTy> {
 public:
  virtual ~BinaryOpHandler() = default;

  SmallVector<Value>* handleOp(OperandRange operands, Value grad,
                               PatternRewriter& rewriter) override {
    SmallVector<Value>* contributions = new SmallVector<Value>();
    this->lhs = operands[0];
    this->rhs = operands[1];
    this->grad = grad;

    auto lhsGrad = getLhsGrad(rewriter);
    auto rhsGrad = getRhsGrad(rewriter);
    contributions->push_back(lhsGrad);
    contributions->push_back(rhsGrad);
    return contributions;
  }

  virtual Value getLhsGrad(PatternRewriter& rewriter) = 0;
  virtual Value getRhsGrad(PatternRewriter& rewriter) = 0;

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
  Value getLhsGrad(PatternRewriter& rewriter) override {
    return oneslike(lhs, rewriter);
  }

  Value getRhsGrad(PatternRewriter& rewriter) override {
    return oneslike(rhs, rewriter);
  }

  static OpHandler& instance() {
    static std::unique_ptr<AddOpHandler> instance(new AddOpHandler());
    return *instance;
  }

 private:
  Value oneslike(Value input, PatternRewriter& rewriter) {
    return createOp<ad::OneslikeOp>(rewriter, input);
  }
};

//===----------------------------------------------------------------------===//
// SubOpHandler
//===----------------------------------------------------------------------===//
class SubOpHandler : public BinaryOpHandler<tosa::SubOp> {
 public:
  Value getLhsGrad(PatternRewriter& rewriter) override {
    return oneslike(lhs, rewriter);
  }

  Value getRhsGrad(PatternRewriter& rewriter) override {
    auto ones = oneslike(rhs, rewriter);
    return rewriter.create<tosa::NegateOp>(rewriter.getUnknownLoc(),
                                           ones.getType(), ones);
  }

  static OpHandler& instance() {
    static std::unique_ptr<SubOpHandler> instance(new SubOpHandler());
    return *instance;
  }

 private:
  Value oneslike(Value input, PatternRewriter& rewriter) {
    return createOp<ad::OneslikeOp>(rewriter, input);
  }
};

//===----------------------------------------------------------------------===//
// MulOpHandler
//===----------------------------------------------------------------------===//
class MulOpHandler : public BinaryOpHandler<tosa::MulOp> {
 public:
  Value getLhsGrad(PatternRewriter& rewriter) override {
    auto type = lhs.getType();
    auto shift = noShift(rewriter);
    return createOp<tosa::MulOp>(rewriter, type, rhs, grad, shift);
  }

  Value getRhsGrad(PatternRewriter& rewriter) override {
    auto type = lhs.getType();
    auto shift = noShift(rewriter);
    return createOp<tosa::MulOp>(rewriter, type, lhs, grad, shift);
  }

  static OpHandler& instance() {
    static std::unique_ptr<MulOpHandler> instance(new MulOpHandler());
    return *instance;
  }

 private:
  IntegerAttr noShift(PatternRewriter& rewriter) {
    return rewriter.getI32IntegerAttr(0);
  }
};

//===----------------------------------------------------------------------===//
// LogOpHandler
//===----------------------------------------------------------------------===//
class LogOpHandler : public UnaryOpHandler<tosa::LogOp> {
 public:
  Value getGrad(PatternRewriter& rewriter) override {
    return createOp<tosa::ReciprocalOp>(rewriter, input.getType(), input);
  }

  static OpHandler& instance() {
    static std::unique_ptr<LogOpHandler> instance(new LogOpHandler());
    return *instance;
  }
};

//===----------------------------------------------------------------------===//
// HandlerFactory
//===----------------------------------------------------------------------===//
class HandlerFactory {
 public:
  static SmallVector<Value>* getResults(Operation* op, Value grad,
                                        PatternRewriter& rewriter) {
    // TODO: how to support extensions
    if (isa<tosa::AddOp>(*op)) {
      return AddOpHandler::instance().handleOp(op, grad, rewriter);
    } else if (isa<tosa::SubOp>(*op)) {
      return SubOpHandler::instance().handleOp(op, grad, rewriter);
    } else if (isa<tosa::MulOp>(*op)) {
      return MulOpHandler::instance().handleOp(op, grad, rewriter);
    } else if (isa<tosa::LogOp>(*op)) {
      return LogOpHandler::instance().handleOp(op, grad, rewriter);
    } else {
      exit(1 && "Unsupported operation detected");
    }
  }
};

}  // namespace mlir::autodiff