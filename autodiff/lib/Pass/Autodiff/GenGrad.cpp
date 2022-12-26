#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace autodiff {

class GenGradPass : public GenGradPassBase<GenGradPass> {
  const StringRef REQGRAD = "requires_grad";
  std::map<int64_t, Value> cache;

  void setOpAttr(OpBuilder& builder, Operation* op) {
    if (op->hasAttr(REQGRAD) || isa<func::ReturnOp>(op)) {
      return;
    }

    auto attr = builder.getI64IntegerAttr(counter());
    op->setAttr(REQGRAD, attr);
  }

  void setGraphAttr(OpBuilder& builder, Operation* op) {
    setOpAttr(builder, op);
    for (auto user : op->getUsers()) {
      setGraphAttr(builder, user);
    }
  }

  void backprop(OpBuilder& builder, Operation* op, Value dout) {
    if (!op || !op->hasAttr(REQGRAD)) {
      return;
    }

    auto index = op->getAttrOfType<IntegerAttr>(REQGRAD).getInt();

    auto loc = builder.getUnknownLoc();
    auto reduce = builder.create<ad::ReduceOp>(loc, dout, cache[index]);
    cache[index] = sum(builder, cache[index], reduce);

    auto opAttr = builder.getStringAttr(op->getName().getStringRef());
    auto inputs = op->getNumOperands();

    if (1 == inputs) {
      auto x = op->getOperand(0);
      auto unary = createOp<grad::AbstractUnaryOp>(builder, x.getType(), x,
                                                   dout, opAttr);

      for (auto attr : op->getAttrs()) {
        unary->setAttr(attr.getName(), attr.getValue());
      }

      auto xOp = getRelatedOperation(x);
      backprop(builder, xOp, unary.getDx());
    } else if (2 == inputs) {
      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      auto binary = createOp<grad::AbstractBinaryOp>(
          builder, lhs.getType(), rhs.getType(), lhs, rhs, dout, opAttr);

      for (auto attr : op->getAttrs()) {
        binary->setAttr(attr.getName(), attr.getValue());
      }

      auto lhsOp = getRelatedOperation(lhs);
      auto rhsOp = getRelatedOperation(rhs);

      backprop(builder, lhsOp, binary.getDlhs());
      backprop(builder, rhsOp, binary.getDrhs());
    } else {
      SmallVector<Value> operands(op->getOperands());
      operands.emplace_back(dout);

      auto grad =
          builder.create<grad::Conv2DOp>(loc, op->getOperandTypes(), operands);
      for (auto attr : op->getAttrs()) {
        grad->setAttr(attr.getName(), attr.getValue());
      }
      for (size_t i = 0; i < grad->getNumOperands(); i++) {
        auto operand = grad->getOperand(i);
        auto owner = getRelatedOperation(operand);
        auto derivative = grad->getResult(i);
        backprop(builder, owner, derivative);
      }
    }
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](func::FuncOp func) {
      auto forwardInputs = func.getArguments();

      auto returnOp = &*func.rbegin()->rbegin();
      auto forwardOutputs = returnOp->getOperands();
      builder.setInsertionPoint(returnOp);

      // replace arguments with placeholders
      for (auto input : forwardInputs) {
        builder.setInsertionPointAfterValue(input);
        auto placeholder = createOp<ad::PlaceholderOp>(builder, input);
        input.replaceAllUsesExcept(placeholder, placeholder);

        // set requires_grad flag
        setGraphAttr(builder, placeholder);
      }

      func.getBody().walk([&](Operation* op) {
        if (auto attr = op->getAttrOfType<IntegerAttr>(REQGRAD)) {
          auto index = attr.getInt();
          auto value = getRelatedValue(op);
          builder.setInsertionPointAfterValue(value);
          cache[index] = zeros(builder, value);
        }
      });

      // backprop
      for (auto output : forwardOutputs) {
        auto op = getRelatedOperation(output);
        if (!op || !op->hasAttr(REQGRAD)) {
          continue;
        }

        auto index = func.getNumArguments();
        func.insertArgument(index, output.getType(), {}, func->getLoc());
        auto dout = func.getArgument(index);
        backprop(builder, op, dout);
      }

      // update return values
      SmallVector<Value> returnValues;
      returnValues.resize(forwardInputs.size());

      func.getBody().walk([&](ad::PlaceholderOp op) {
        auto argIndex = op.getInput().cast<BlockArgument>().getArgNumber();
        auto cacheIndex = op->getAttrOfType<IntegerAttr>(REQGRAD).getInt();
        returnValues[argIndex] = cache[cacheIndex];
      });

      returnOp->insertOperands(returnOp->getNumOperands(), returnValues);

      // update function signature
      auto symName = func.getSymName();
      auto newName = ("diff_" + symName).str();
      func.setSymName(newName);

      auto argsType = func.getArgumentTypes();
      auto newType =
          builder.getFunctionType(argsType, returnOp->getOperandTypes());
      func.setFunctionType(newType);

      // remove unused attribute
      func.getBody().walk([&](Operation* op) { op->removeAttr(REQGRAD); });
    });
  }
};

std::unique_ptr<Pass> createADGenGradPass() {
  return std::make_unique<GenGradPass>();
}

}  // namespace autodiff
}  // namespace mlir
