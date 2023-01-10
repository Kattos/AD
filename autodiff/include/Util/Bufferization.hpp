#ifndef AD_UTIL_BUFFERIZATION_HPP
#define AD_UTIL_BUFFERIZATION_HPP

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
namespace autodiff {
namespace util {
namespace bufferization {
/**
 * @brief alloc a tensor that has the same shape and element type as the input
 * value
 *
 * @param value
 * @param builder
 * @return Value
 */
Value alloc(Value value, OpBuilder& builder);
}  // namespace bufferization
}  // namespace util
}  // namespace autodiff
}  // namespace mlir

#endif  // AD_UTIL_BUFFERIZATION_HPP
