#include "Util/Generic.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "Util/Arith.hpp"
#include "Util/Bufferization.hpp"
#include "Util/Tape.hpp"
#include "mlir/IR/BlockAndValueMapping.h"

namespace mlir {
namespace autodiff {
namespace util {
namespace generic {

linalg::GenericOp Reverser::reverse(OpBuilder& builder, Value dout) {
  auto loc = builder.getUnknownLoc();

  auto forwardInputs = forward.getInputs();
  auto reverseInputs = SmallVector<Value>(forwardInputs);
  reverseInputs.emplace_back(dout);

  SmallVector<Value> reverseOutputs;
  SmallVector<Type> reverseTypes;

  llvm::transform(
      forwardInputs, std::back_inserter(reverseOutputs),
      [&](Value value) { return util::bufferization::alloc(value, builder); });

  llvm::transform(forwardInputs, std::back_inserter(reverseTypes),
                  [&](Value value) { return value.getType(); });

  auto forwardMaps = forward.getIndexingMapsArray();
  auto reverseMaps = SmallVector<AffineMap>(forwardMaps);
  std::transform(forwardMaps.begin(), forwardMaps.end() - 1,
                 std::back_inserter(reverseMaps),
                 [](AffineMap map) { return map; });

  auto reverseIters = forward.getIteratorTypesArray();

  auto forwardBody = forward.getBody();

  auto reverseBody = [&](OpBuilder& builder, Location loc, ValueRange args) {
    BlockAndValueMapping mapping;
    mapping.map(forwardBody->getArguments(),
                args.take_front(reverseInputs.size()));

    SmallVector<Operation*> cloned;

    using Ty = decltype(*forwardBody->begin());
    llvm::transform(*forwardBody, std::back_inserter(cloned),
                    [&](Ty op) { return builder.clone(op, mapping); });

    auto yield = cloned.back();
    auto outputs = yield->getOperands();
    auto inputs = args.take_front(forwardInputs.size());
    auto tape = util::tape::record(inputs, outputs, builder);

    yield->erase();
    SmallVector<Value> grads;

    for (auto in : inputs) {
      for (auto out : outputs) {
        auto grad =
            builder.create<grad::GradientOp>(loc, in.getType(), out, in);
        grads.emplace_back(grad);
      }
    }

    auto dout = args[forwardInputs.size()];
    SmallVector<Value> yields;
    for (auto pair : llvm::zip(grads, args.take_back(grads.size()))) {
      auto add =
          util::arith::add(std::get<0>(pair), std::get<1>(pair), builder);
      auto yield = util::arith::mul(dout, add, builder);
      yields.emplace_back(yield);
    }

    builder.create<linalg::YieldOp>(loc, yields);
  };

  return builder.create<linalg::GenericOp>(loc, reverseTypes, reverseInputs,
                                           reverseOutputs, reverseMaps,
                                           reverseIters, reverseBody);
}

}  // namespace generic
}  // namespace util
}  // namespace autodiff
}  // namespace mlir
