#include "Conversion/GradToCore/GradToCore.hpp"
#include "Util/Tape.hpp"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

using namespace util;

class GradientToCore : public OpRewritePattern<grad::GradientOp> {
  using OpRewritePattern<grad::GradientOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::GradientOp gradient,
                                PatternRewriter& rewriter) const override {
    auto target = gradient.target();
    auto source = gradient.source();

    auto tape = tape::record(source, target, rewriter);
    rewriter.replaceOp(gradient, tape.get(source, rewriter));

    return success();
  }
};

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
