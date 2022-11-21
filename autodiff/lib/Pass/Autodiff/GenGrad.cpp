#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    if (!op) {
      return;
    }

    auto index = op->getAttrOfType<IntegerAttr>(REQGRAD).getInt();
    cache[index] = sum(builder, cache[index], dout);

    auto opAttr = builder.getStringAttr(op->getName().getStringRef());
    auto inputs = op->getNumOperands();

    if (1 == inputs) {
      auto x = op->getOperand(0);
      auto unary = createOp<grad::AbstractUnaryOp>(builder, x.getType(), x,
                                                   dout, opAttr);

      auto xOp = getRelatedOperation(x);
      backprop(builder, xOp, unary.getDx());
    } else if (2 == inputs) {
      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      auto binary = createOp<grad::AbstractBinaryOp>(
          builder, lhs.getType(), rhs.getType(), lhs, rhs, dout, opAttr);

      auto lhsOp = getRelatedOperation(lhs);
      auto rhsOp = getRelatedOperation(rhs);

      backprop(builder, lhsOp, binary.getDlhs());
      backprop(builder, rhsOp, binary.getDrhs());
    }
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](func::FuncOp func) {
      // replace arguments with placeholders
      for (auto argument : func.getArguments()) {
        builder.setInsertionPointAfterValue(argument);
        auto placeholder = createOp<ad::PlaceholderOp>(builder, argument);
        argument.replaceAllUsesExcept(placeholder, placeholder);

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
      auto returnOp = &*func.rbegin()->rbegin();
      auto outputs = returnOp->getOperands();
      builder.setInsertionPoint(returnOp);

      for (auto output : outputs) {
        auto op = getRelatedOperation(output);
        if (!op || !op->hasAttr(REQGRAD)) {
          continue;
        }

        auto dout = ones(builder, output);
        backprop(builder, op, dout);
      }

      // update function signature
      auto symName = func.getSymName();
      auto newName = ("diff_" + symName).str();
      func.setSymName(newName);

      auto argsType = func.getArgumentTypes();
      auto newType = builder.getFunctionType(argsType, argsType);
      func.setFunctionType(newType);

      // update return values
      returnOp->eraseOperands(0, returnOp->getNumOperands());

      SmallVector<Value> returnValues;
      returnValues.resize(func.getNumArguments());

      func.getBody().walk([&](ad::PlaceholderOp op) {
        auto argIndex = op.getInput().cast<BlockArgument>().getArgNumber();
        auto cacheIndex = op->getAttrOfType<IntegerAttr>(REQGRAD).getInt();
        returnValues[argIndex] = cache[cacheIndex];
      });

      returnOp->setOperands(returnValues);
    });
  }
};

std::unique_ptr<Pass> createADGenGradPass() {
  return std::make_unique<GenGradPass>();
}

}  // namespace autodiff
}  // namespace mlir
