#ifndef AD_UTIL_GENERIC_HPP
#define AD_UTIL_GENERIC_HPP

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace autodiff {
namespace util {
namespace generic {

class Reverser {
 private:
  linalg::GenericOp forward;

 public:
  Reverser(linalg::GenericOp forward) : forward(forward) {}
  linalg::GenericOp reverse(OpBuilder& builder, Value dout);
};

}  // namespace generic
}  // namespace util
}  // namespace autodiff
}  // namespace mlir

#endif  // AD_UTIL_GENERIC_HPP
