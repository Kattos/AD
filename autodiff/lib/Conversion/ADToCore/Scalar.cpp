#include "Conversion/ADToCore/ADToCore.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace autodiff {

class ScalarTensorToCore : public OpRewritePattern<ad::ScalarTensorOp> {
  using OpRewritePattern<ad::ScalarTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::ScalarTensorOp op,
                                PatternRewriter& rewriter) const override {
    auto x = op.getInput();
    auto tensor = createOp<tensor::FromElementsOp>(rewriter, op.getType(), x);
    rewriter.replaceOp(op, tensor.getResult());
    return success();
  }
};

}  // namespace autodiff
}  // namespace mlir
