#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradInterface.hpp"
#include "GenericWrapper.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace autodiff {
class ExperimentalPass : public ExperimentalPassBase<ExperimentalPass> {
 public:
  void runOnOperation() override {
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
      if (auto adjoint = dyn_cast<AdjointInterface>(op)) {
        auto adjoints = adjoint.adjoint(builder);
        for (auto value : adjoints) {
          value.dump();
        }
      }
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
};

std::unique_ptr<Pass> createExperimentalPass() {
  return std::make_unique<ExperimentalPass>();
}

}  // namespace autodiff
}  // namespace mlir
