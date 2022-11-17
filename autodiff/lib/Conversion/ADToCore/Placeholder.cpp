#include "Conversion/Conversion.hpp"

namespace mlir::autodiff {

class PlaceholderToCore : public OpRewritePattern<ad::PlaceholderOp> {
  using OpRewritePattern<ad::PlaceholderOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ad::PlaceholderOp placeholder,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(placeholder, placeholder.getInput());
    return success();
  }
};

}  // namespace mlir::autodiff
