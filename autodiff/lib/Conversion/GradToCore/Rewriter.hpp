#ifndef GRADTOCORE_REWRITER_H
#define GRADTOCORE_REWRITER_H

#include "Conversion/Conversion.hpp"
#include "Rule/Helper.hpp"
#include "Rule/Rules.hpp"
#include "Rule/Utils.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

template <typename GradTy>
Value unaryCalFn(Operation* op, ValueRange args, ArrayRef<Type> resultTypes,
                 PatternRewriter& rewriter) {
  auto x = args[0];
  auto dout = args[1];

  auto gradOp = createOp<GradTy>(rewriter, x.getType(), x);
  auto grad = getGradient(gradOp, ones(rewriter, gradOp), x);

  return product(rewriter, dout, grad);
}

template <typename OpTy, typename GradTy>
class UnaryToCore : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();

    if (auto shapedType = resultType.template dyn_cast<ShapedType>())
      return elementwiseMatchAndRewriteHelper(op, rewriter, unaryCalFn<GradTy>);

    auto value = unaryCalFn<GradTy>(op, op.getOperands(), resultType, rewriter);

    if (!value) return failure();

    rewriter.replaceOp(op, value);
    return success();
  }
};

using RsqrtToCore = UnaryToCore<grad::RsqrtOp, math::RsqrtOp>;
using LogToCore = UnaryToCore<grad::LogOp, math::LogOp>;
using ExpToCore = UnaryToCore<grad::ExpOp, math::ExpOp>;
using TanhToCore = UnaryToCore<grad::TanhOp, math::TanhOp>;

}  // namespace mlir::autodiff

#endif  // GRADTOCORE_REWRITER_H
