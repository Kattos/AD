#include <iterator>

#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dual.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace autodiff {

using linalg::GenericOp;

class ReverseGeneric {
 public:
  class Builder {
   private:
    using BodyFn = function_ref<void(OpBuilder&, Location, ValueRange)>;
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;
    SmallVector<Type> types;
    SmallVector<AffineMap> maps;
    SmallVector<StringRef> iters;
    BodyFn fn;

   public:
    GenericOp build(OpBuilder& builder) {
      return builder.create<linalg::GenericOp>(
          builder.getUnknownLoc(), types, inputs, outputs, maps, iters, fn);
    }

    template <typename T>
    static void clear(SmallVector<T>& vector, size_t newSize) {
      vector.clear();
      vector.reserve(newSize);
    }

    template <typename FromTy, typename ToTy>
    static void fill(SmallVector<FromTy> froms, SmallVector<ToTy>& tos,
                     function_ref<ToTy(FromTy)> fillFn, size_t size = 0) {
      if (size == 0) {
        size = froms.size();
      }

      for (size_t i = 0; i < size; ++i) {
        tos.emplace_back(fillFn(froms[i]));
      }
    }

    Builder reverseInputs(GenericOp op, Value dout) {
      auto opInputs = op.getInputs();

      clear(inputs, opInputs.size() + 1);
      fill<Value, Value>(opInputs, inputs, [](Value v) { return v; });
      inputs.emplace_back(dout);

      return *this;
    }

    Builder reverseOutputs(GenericOp op, OpBuilder& builder) {
      auto opInputs = op.getInputs();

      clear(outputs, opInputs.size());
      fill<Value, Value>(opInputs, outputs, [&](Value v) {
        return builder.create<ad::ZeroslikeOp>(builder.getUnknownLoc(), v);
      });

      return *this;
    }

    Builder reverseTypes(GenericOp op) {
      auto opInputs = op.getInputs();

      clear(types, opInputs.size());
      fill<Value, Type>(opInputs, types, [](Value v) { return v.getType(); });

      return *this;
    }

    Builder reverseMaps(GenericOp op) {
      auto opMaps = op.getIndexingMapsArray();

      clear(maps, opMaps.size() * 2 - 1);
      auto fillFn = [](AffineMap map) { return map; };
      fill<AffineMap, AffineMap>(opMaps, maps, fillFn);
      fill<AffineMap, AffineMap>(opMaps, maps, fillFn, opMaps.size() - 1);

      return *this;
    }

    Builder reverseIters(GenericOp op) {
      auto opIters = op.getIteratorTypes();

      // clear(iters, opIters.size());
      // fill<StringRef, StringRef>(opIters, iters,
      //                            [](StringRef iter) { return iter; });

      llvm::transform(opIters, std::back_inserter(iters),
                      [](decltype(*opIters.begin()) i) {
                        return i.dyn_cast<StringAttr>().getValue();
                      });

      return *this;
    }

    // TODO: does it need a new grad dialect here?
    Builder reverseBody(GenericOp op) {
      // TODO: backprop on arith/math ops here
      return *this;
    }
  };

 private:
  GenericOp primal;
  Builder builder;

 public:
  ReverseGeneric(const GenericOp primal) : primal(primal) {
    builder = Builder();
  }

  GenericOp reverse(OpBuilder& ob, Value dout) {
    return builder.reverseInputs(primal, dout)
        .reverseOutputs(primal, ob)
        .reverseTypes(primal)
        .reverseMaps(primal)
        .reverseIters(primal)
        .reverseBody(primal)
        .build(ob);
  }
};

class GradGenericPass : public GradGenericPassBase<GradGenericPass> {
  void runOnOperation() override { llvm::outs() << "GradGenericPass\n"; }
};

std::unique_ptr<Pass> createADGradGenericPass() {
  return std::make_unique<GradGenericPass>();
}

}  // namespace autodiff
}  // namespace mlir
