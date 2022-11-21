#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
#include "Rule/Utils.hpp"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace autodiff {

class AbstractBinaryToConcrete
    : public OpRewritePattern<grad::AbstractBinaryOp> {
  using OpRewritePattern<grad::AbstractBinaryOp>::OpRewritePattern;

  template <typename OpTy>
  ValueRange lower(grad::AbstractBinaryOp binary,
                   PatternRewriter& rewriter) const {
    auto resultTypes = binary->getResultTypes();
    auto operands = binary->getOperands();

    binary->removeAttr("op");
    auto attrs = binary->getAttrs();

    return createOp<OpTy>(rewriter, resultTypes, operands, attrs).getResults();
  }

  LogicalResult matchAndRewrite(grad::AbstractBinaryOp binary,
                                PatternRewriter& rewriter) const override {
    auto op = binary.getOp();
    ValueRange concrete;

    if (op == tosa::AddOp::getOperationName())
      concrete = lower<grad::AddOp>(binary, rewriter);

    if (concrete.empty()) {
      binary->emitError() << "Failed to rewrite `grad.abstract_binary`";
      return failure();
    }

    rewriter.replaceOp(binary, concrete);
    return success();
  }
};

}  // namespace autodiff
}  // namespace mlir
