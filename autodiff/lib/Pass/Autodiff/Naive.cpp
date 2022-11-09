#include <cstdio>

#include "ADUtils.hpp"
#include "Dialect/AD/IR/AD.hpp"
#include "OpHandler.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {

class NaivePass : public NaivePassBase<NaivePass> {
  void initFunc(func::FuncOp func) {
    OpBuilder builder(func.getBody());
    auto loc = func->getLoc();

    // insert `to` ops
    for (auto argument : func.getArguments()) {
      auto to = builder.create<ad::ToOp>(loc, argument);
      // ops use `to`'s result instead
      argument.replaceAllUsesExcept(to, to);
    }

    // `return` must be last op of `func`
    auto returnOp = func.rbegin()->rbegin();

    // insert `from` ops
    builder.setInsertionPoint(&*returnOp);
    for (auto operand : returnOp->getOperands()) {
      builder.create<ad::FromOp>(loc, operand);
    }

    // change function type
    auto argsType = func.getArgumentTypes();
    auto funcType = builder.getFunctionType(argsType, argsType);
    func.setFunctionType(funcType);

    // change function name
    auto newName = ("diff_" + func.getSymName()).str();
    func.setSymName(newName);

    // replace operands of `return`
    returnOp->setOperands(func.getArguments());
  }

  void initIndex(func::FuncOp func) {
    OpBuilder builder(func.getBody());

    func.getBody().walk([&](Operation* op) {
      if (isa<func::ReturnOp>(*op)) {
        return;
      }

      auto attr = builder.getI64IntegerAttr(counter());
      op->setAttr(INDEX, attr);
    });
  }

  void initGrad(func::FuncOp func) {
    OpBuilder builder(func.getBody());
    auto loc = func->getLoc();

    func.getBody().walk([&](ad::FromOp from) {
      builder.setInsertionPoint(from);
      auto ones = builder.create<ad::OneslikeOp>(loc, from.getInput());
      auto index = from->getAttrOfType<IntegerAttr>(INDEX).getInt();
      indexGradMap[index] = ones;
    });
  }

  void init(func::FuncOp func) {
    initFunc(func);
    initIndex(func);
    initGrad(func);
  }

  void genGrad(ad::ToOp to, func::FuncOp func) {
    auto returnOp = func.rbegin()->rbegin();
    OpBuilder builder(&*returnOp);

    // 1. evaluate grad for `to` op
    auto grad = evaluate(to, builder);

    // 2. replace `return` operand with `to` grad
    for (size_t i = 0; i < returnOp->getOperands().size(); ++i) {
      if (to.getInput() == returnOp->getOperands()[i]) {
        returnOp->setOperand(i, grad);
      }
    }

    // 3. replace users operand and erase `to`
    to.getOutput().replaceAllUsesWith(to.getInput());
    to.erase();
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](func::FuncOp func) {
      init(func);

      func.getBody().walk([&](ad::ToOp to) { genGrad(to, func); });

      func.getBody().walk([&](Operation* op) {
        op->removeAttr(INDEX);
        if (isa<ad::FromOp>(*op)) {
          op->erase();
        }
      });
    });
  }

  const StringRef INDEX = "index";

  int64_t counter() {
    static int64_t index = 0;
    return ++index;
  }

  std::map<int64_t, Value> indexGradMap;

  Value evaluate(Operation* op, OpBuilder& builder) {
    if (!op || !op->hasAttr(INDEX)) {
      return nullptr;
    }

    auto index = op->getAttrOfType<IntegerAttr>(INDEX).getInt();
    if (indexGradMap[index]) {
      return indexGradMap[index];
    }

    Value grad = nullptr;
    Value value = getRelatedValue(op);

    for (auto user : op->getUsers()) {
      if (!user->hasAttr(INDEX)) {
        continue;
      }

      Value contribution = nullptr;
      if (isa<ad::FromOp>(*user)) {
        auto userIndex = user->getAttrOfType<IntegerAttr>(INDEX).getInt();
        contribution = indexGradMap[userIndex];
      } else {
        auto userGrad = evaluate(user, builder);

        contribution =
            HandlerFactory::getContribution(user, userGrad, value, builder);
      }

      grad = sum(builder, grad, contribution);
    }

    indexGradMap[index] = grad;
    return grad;
  }
};

std::unique_ptr<Pass> createADNaivePass() {
  return std::make_unique<NaivePass>();
}

}  // namespace mlir::autodiff