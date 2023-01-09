#include "Util/Tape.hpp"

#include "Dialect/Grad/IR/GradInterface.hpp"
#include "Util/Arith.hpp"

namespace mlir {
namespace autodiff {
namespace util {
namespace tape {

void Path::push(Pair pair) { pairs.emplace_back(pair); }
Pair Path::pop() { return pairs.pop_back_val(); }

Value Path::evaluate(OpBuilder& builder) {
  using It = decltype(pairs.rbegin());
  using Fn = function_ref<Value(It, It, Value, OpBuilder&)>;

  Fn recursive = [&recursive](It curr, It end, Value gradient,
                              OpBuilder& builder) {
    if (curr == end) {
      return gradient;
    }

    auto [in, op] = *curr;

    for (auto i = 0u; i < op->getNumOperands(); i++) {
      if (in == op->getOperand(i)) {
        auto partial = dyn_cast<PartialInterface>(op).partialFor(builder, i);
        gradient = arith::mul(gradient, partial, builder);
        break;
      }
    }

    return recursive(++curr, end, gradient, builder);
  };

  auto initial = arith::constant(1.0, builder);
  return recursive(pairs.rbegin(), pairs.rend(), initial, builder);
}

SmallVector<Path> navigate(Value from, Value to) {
  using Fn = function_ref<void(Value, Value, Path&)>;

  SmallVector<Path> paths;
  Path path;

  Fn dfs = [&dfs, &paths](Value from, Value to, Path& path) {
    if (from == to) {
      paths.emplace_back(path);
      return;
    }

    for (auto user : from.getUsers()) {
      path.push(std::make_pair(from, user));
      for (auto result : user->getResults()) {
        dfs(result, to, path);
      }
      path.pop();
    }
  };

  dfs(from, to, path);
  return paths;
}

Value Tape::get(Value in) { return tape[in]; }

Value Tape::get(Value in, OpBuilder& builder) {
  return tape[in] ? tape[in] : arith::constant(0.0, builder);
}

Tape record(ValueRange ins, ValueRange outs, OpBuilder& builder) {
  Tape instance;
  for (auto in : ins) {
    for (auto out : outs) {
      auto paths = navigate(in, out);
      for (auto path : paths) {
        instance.tape[in] =
            instance.tape[in]
                ? arith::add(instance.tape[in], path.evaluate(builder), builder)
                : path.evaluate(builder);
      }
    }
  }
  return instance;
}

}  // namespace tape
}  // namespace util
}  // namespace autodiff
}  // namespace mlir
