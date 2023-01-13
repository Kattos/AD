#ifndef AD_UTIL_OTHERS_HPP
#define AD_UTIL_OTHERS_HPP

#include "mlir/IR/Operation.h"

namespace mlir {
namespace autodiff {
namespace util {
namespace others {

std::optional<int> indexOfOperand(Operation* op, Value value);

}  // namespace others
}  // namespace util
}  // namespace autodiff
}  // namespace mlir

#endif  // AD_UTIL_OTHERS_HPP
