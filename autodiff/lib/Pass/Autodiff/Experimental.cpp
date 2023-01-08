#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradInterface.hpp"
#include "GenericWrapper.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Util/Tape.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace autodiff {

class ExperimentalPass : public ExperimentalPassBase<ExperimentalPass> {
 public:
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);
    // auto loc = builder.getUnknownLoc();

    module->walk([&](func::FuncOp func) {
      auto inputs = func.getArguments();
      auto returnOp = &*func.front().rbegin();
      auto outputs = returnOp->getOperands();

      builder.setInsertionPoint(returnOp);
      auto tape = util::tape::Tape::record(inputs, outputs, builder);
      SmallVector<Value> newOutputs;
      for (auto input : inputs) {
        newOutputs.emplace_back(tape.get(input));
      }
      returnOp->setOperands(newOutputs);

      auto funcType = func.getFunctionType();
      auto newFuncType =
          builder.getFunctionType(funcType.getInputs(), funcType.getInputs());
      func.setFunctionType(newFuncType);
    });
  }

 private:
  void nablaFunction() {
    auto module = getOperation();
    OpBuilder builder(module);
    auto loc = builder.getUnknownLoc();

    module->walk([&](func::FuncOp func) {
      builder.setInsertionPointAfter(func);

      auto inputs = func.getFunctionType().getInputs();
      auto outputs = func.getFunctionType().getResults();

      auto revInputs = SmallVector<Type>(inputs);
      std::copy(outputs.begin(), outputs.end(), std::back_inserter(revInputs));

      auto revName = ("nabla_" + func.getSymName()).str();
      auto revType = builder.getFunctionType(revInputs, inputs);
      auto revFunc = builder.create<func::FuncOp>(loc, revName, revType);

      auto body = revFunc.addEntryBlock();
      auto bodyBuilder = OpBuilder::atBlockBegin(body);

      auto fn = FlatSymbolRefAttr::get(func);
      auto rev = bodyBuilder.create<grad::NablaOp>(loc, inputs, fn,
                                                   revFunc.getArguments());
      bodyBuilder.create<func::ReturnOp>(loc, rev.getOutputs());
    });
  }

  void nablaGeneric() {
    auto module = getOperation();
    OpBuilder builder(module);
    auto loc = builder.getUnknownLoc();

    module->walk([&](func::FuncOp func) {
      auto returnOp = func.rbegin()->rbegin();
      builder.setInsertionPoint(&*returnOp);

      linalg::GenericOp reverse = nullptr;

      func->walk([&](linalg::GenericOp generic) {
        auto wrapper = GenericWrapper(generic);
        auto one = builder.create<ad::OneslikeOp>(loc, generic.getResults()[0]);
        reverse = wrapper.reverse(builder, one);
      });

      if (reverse) {
        returnOp->setOperands(reverse.getResults());
        auto type = builder.getFunctionType(func.getArgumentTypes(),
                                            returnOp->getOperandTypes());
        func.setFunctionType(type);
      }
    });

    module->walk([&](Operation* op) {
      if (auto intf = dyn_cast<PartialInterface>(op)) {
        auto derivatives = intf.partial(builder);
      }
    });
  }
};

std::unique_ptr<Pass> createExperimentalPass() {
  return std::make_unique<ExperimentalPass>();
}

}  // namespace autodiff
}  // namespace mlir
