#include "Dialect/AD/IR/AD.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace autodiff {

class GenericWrapper {
 private:
  linalg::GenericOp forward;

 public:
  GenericWrapper(linalg::GenericOp forward) : forward(forward) {}

  linalg::GenericOp reverse(OpBuilder& builder, Value dout) {
    auto loc = builder.getUnknownLoc();

    auto forwardInputs = forward.getInputs();
    auto reverseInputs = SmallVector<Value>(forwardInputs);
    reverseInputs.emplace_back(dout);

    auto reverseOutputs = SmallVector<Value>();
    auto reverseTypes = SmallVector<Type>();

    for (auto input : forwardInputs) {
      auto zero = builder.create<ad::ZeroslikeOp>(loc, input);
      auto type = input.getType();

      reverseOutputs.emplace_back(zero);
      reverseTypes.emplace_back(type);
    }

    auto forwardMaps = forward.getIndexingMapsArray();
    auto reverseMaps = SmallVector<AffineMap>(forwardMaps);
    for (size_t i = 0; i < forwardMaps.size() - 1; i++) {
      reverseMaps.emplace_back(forwardMaps[i]);
    }

    auto reverseIters = forward.getIteratorTypesArray();

    return builder.create<linalg::GenericOp>(loc, reverseTypes, reverseInputs,
                                             reverseOutputs, reverseMaps,
                                             reverseIters, reverseBody());
  }

  function_ref<void(OpBuilder&, Location, ValueRange)> reverseBody() {
    // TODO: backprop
    return [&](OpBuilder& builder, Location loc, ValueRange args) {
      SmallVector<Value> yields;
      for (size_t i = 0; i < forward.getInputs().size(); i++) {
        yields.emplace_back(args[i]);
      }
      builder.create<linalg::YieldOp>(loc, yields);
    };
  }
};

}  // namespace autodiff
}  // namespace mlir
