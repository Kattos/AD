#include "Conversion/GradToCore/GradToCore.hpp"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

SmallVector<int64_t> attrToArray(ArrayAttr attrs, SmallVector<int64_t>& array);
Value pad2DTensor(PatternRewriter& rewriter, Value tensor, ArrayAttr padAttr);
Value unpad2DTensor(PatternRewriter& rewriter, Value paddedTensor,
                    ArrayAttr padAttr);

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
