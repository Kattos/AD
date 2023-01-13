#include "Util/Others.hpp"

namespace mlir {
namespace autodiff {
namespace util {
namespace others {

std::optional<int> indexOfOperand(Operation* op, Value value) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (value == operand) {
      return std::make_optional(index);
    }
  }

  return std::nullopt;
}

}  // namespace others
}  // namespace util
}  // namespace autodiff
}  // namespace mlir
