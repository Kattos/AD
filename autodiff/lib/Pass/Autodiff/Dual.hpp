#include "Dialect/AD/IR/AD.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace autodiff {

template <typename T>
concept IsOp = std::is_base_of_v<OpState, T>;

class DualBase {
 private:
  DualBase() = default;
};

template <typename T>
requires IsOp<T>
class Dual : DualBase {
 private:
  T op;
  Value gradient;
  friend class DualFactory;

 public:
  Dual(T op, Value gradient) : op(op), gradient(gradient) {}
  T operator()() const { return op; }
};

class DualFactory {
 private:
  static llvm::DenseMap<OpState, DualBase> cache;

 public:
  template <typename T>
  requires IsOp<T>
  static Dual<T> get(T op, OpBuilder& builder) {
    if (auto dual = cache.find(op)) {
      return dual;
    }

    auto loc = builder.getUnknownLoc();
    auto zero = builder.create<ad::ZeroslikeOp>(loc, op);
    return get(op, zero);
  }

  template <typename T>
  requires IsOp<T>
  static Dual<T> get(T op, Value gradient) {
    if (auto dual = cache.find(op)) {
      dual.gradient = gradient;
      return dual;
    }

    auto dual = Dual(op, gradient);
    cache[op] = dual;
    return dual;
  }
};

}  // namespace autodiff
}  // namespace mlir
