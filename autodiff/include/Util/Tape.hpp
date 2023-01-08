#ifndef AD_TAPE_HPP
#define AD_TAPE_HPP

#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/Grad/IR/Grad.hpp"

namespace mlir {
namespace autodiff {
namespace util {
namespace tape {

using Node = Value;
using Edge = Operation*;
using Pair = std::pair<Value, Operation*>;

class Path {
 private:
  SmallVector<Pair> pairs;

 public:
  Path() = default;
  void push(Pair pair);
  Pair pop();
  Value evaluate(OpBuilder& builder);
};

class Tape {
 private:
  DenseMap<Value, Value> tape;
  Tape() = default;

 public:
  static Tape record(ValueRange ins, ValueRange outs, OpBuilder& builder);
  Value get(Value in);
};

SmallVector<Path> navigate(Value from, Value to);

}  // namespace tape
}  // namespace util
}  // namespace autodiff
}  // namespace mlir

#endif  // AD_TAPE_HPP
