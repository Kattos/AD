#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
#include "Rule/Utils.hpp"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace autodiff {

class AbstractUnaryToConcrete : public OpRewritePattern<grad::AbstractUnaryOp> {
  using OpRewritePattern<grad::AbstractUnaryOp>::OpRewritePattern;

  template <typename OpTy>
  Value lower(grad::AbstractUnaryOp unary, PatternRewriter& rewriter) const {
    auto resultTypes = unary->getResultTypes();
    auto operands = unary->getOperands();

    unary->removeAttr("op");
    auto attrs = unary->getAttrs();

    return createOp<OpTy>(rewriter, resultTypes, operands, attrs);
  }

  LogicalResult matchAndRewrite(grad::AbstractUnaryOp unary,
                                PatternRewriter& rewriter) const override {
    auto op = unary.getOp();
    Value concrete = nullptr;

    if (op == tosa::AbsOp::getOperationName())
      concrete = lower<grad::AbsOp>(unary, rewriter);

    else if (op == tosa::ExpOp::getOperationName())
      concrete = lower<grad::ExpOp>(unary, rewriter);

    else if (op == tosa::LogOp::getOperationName())
      concrete = lower<grad::LogOp>(unary, rewriter);

    else if (op == tosa::RsqrtOp::getOperationName())
      concrete = lower<grad::RsqrtOp>(unary, rewriter);

    else if (op == tosa::TanhOp::getOperationName())
      concrete = lower<grad::TanhOp>(unary, rewriter);

    else if (op == tosa::ClampOp::getOperationName())
      concrete = lower<grad::ClampOp>(unary, rewriter);

    else if (op == tosa::NegateOp::getOperationName())
      concrete = lower<grad::ClampOp>(unary, rewriter);

    else if (op == tosa::ReciprocalOp::getOperationName())
      concrete = lower<grad::ClampOp>(unary, rewriter);

    else if (op == tosa::SigmoidOp::getOperationName())
      concrete = lower<grad::SigmoidOp>(unary, rewriter);

    if (!concrete) {
      unary->emitOpError() << "Failed to rewrite `grad.abstract_unary`";
      return failure();
    }

    rewriter.replaceOp(unary, concrete);
    return success();
  }
};

}  // namespace autodiff
}  // namespace mlir
