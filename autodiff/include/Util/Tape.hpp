#ifndef AD_UTIL_TAPE_HPP
#define AD_UTIL_TAPE_HPP

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
  friend Tape record(ValueRange ins, ValueRange outs, OpBuilder& builder);

 private:
  DenseMap<Value, Value> tape;
  Tape() = default;

 public:
  Value get(Value in);
  Value get(Value in, OpBuilder& builder);
};

SmallVector<Path> navigate(Value from, Value to);
Tape record(ValueRange ins, ValueRange outs, OpBuilder& builder);

}  // namespace tape
}  // namespace util
}  // namespace autodiff
}  // namespace mlir

#endif  // AD_UTIL_TAPE_HPP
