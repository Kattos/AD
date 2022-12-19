#include "Conversion/GradToCore/GradToCore.hpp"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

Value dReduceSum(PatternRewriter& rewriter, Value output) {
  auto reduce = output.getDefiningOp<grad::ReduceSumOp>();
  if (!reduce) {
    return nullptr;
  }

  auto loc = rewriter.getUnknownLoc();

  auto x = reduce.getX();
  auto dout = reduce.getDout();

  return nullptr;
}

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
