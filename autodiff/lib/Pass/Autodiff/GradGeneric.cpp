#include "Dialect/AD/IR/AD.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace autodiff {

class GradGenericPass : public GradGenericPassBase<GradGenericPass> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);
    auto loc = builder.getUnknownLoc();

    module->walk([&](func::FuncOp func) {
      func->walk([&](linalg::GenericOp generic) {
        if (isa<linalg::YieldOp>(generic.getBlock()->front())) {
          return;
        }

        generic->setAttr("requires_grad", builder.getBoolAttr(true));
      });

      auto returnOp = func.rbegin()->rbegin();
      builder.setInsertionPoint(&*returnOp);

      func->walk([&](linalg::GenericOp generic) {
        if (!generic->hasAttr("requires_grad")) {
          return;
        }

        auto genericInputs = SmallVector<Value>(generic.getInputs());
        auto genericInputsSize = genericInputs.size();

        // TODO: support for multiple outputs
        auto genericOutput = generic.getOutputs()[0];
        auto genericMaps = generic.getIndexingMapsArray();
        auto genericIters = generic.getIteratorTypesArray();

        // TODO: get from user inputs
        auto dout = builder.create<ad::OneslikeOp>(loc, genericOutput);
        genericInputs.emplace_back(dout);
        auto inputs = genericInputs;

        SmallVector<Value> outputs;
        outputs.reserve(genericInputs.size());

        // TODO: get from a map or something
        for (auto input : generic.getInputs()) {
          outputs.emplace_back(builder.create<ad::ZeroslikeOp>(loc, input));
        }

        SmallVector<Type> types;
        types.reserve(outputs.size());
        for (auto output : outputs) {
          types.emplace_back(output.getType());
        }

        auto iterTypes = genericIters;
        SmallVector<AffineMap> idxMaps;
        idxMaps.reserve(genericInputsSize * 2 + 1);
        for (size_t i = 0; i < genericInputsSize; ++i) {
          idxMaps.emplace_back(genericMaps[i]);
        }
        idxMaps.emplace_back(genericMaps[genericMaps.size() - 1]);
        for (size_t i = 0; i < genericInputsSize; ++i) {
          idxMaps.emplace_back(genericMaps[i]);
        }

        auto calculator = [&](OpBuilder& builder, Location loc,
                              ValueRange args) {
          // TODO: evaluate gradients here
          builder.create<linalg::YieldOp>(loc, args.take_front(outputs.size()));
        };

        auto linalg = builder.create<linalg::GenericOp>(
            loc, types, inputs, outputs, idxMaps, iterTypes, calculator);
      });
    });
  }
};

std::unique_ptr<Pass> createADGradGenericPass() {
  return std::make_unique<GradGenericPass>();
}

}  // namespace autodiff
}  // namespace mlir
