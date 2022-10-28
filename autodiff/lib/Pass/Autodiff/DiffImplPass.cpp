#include "Pass/Autodiff/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::autodiff {

class LogOpConversionPattern : public OpConversionPattern<tosa::LogOp> {
  using OpConversionPattern<tosa::LogOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tosa::LogOp op, tosa::LogOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<tosa::ExpOp>(op, op.getOutput().getType(),
                                             op.getInput1());
    return success();
  }
};

class DiffImplPass : public DiffImplPassBase<DiffImplPass> {
  void runOnOperation() override {}
};

std::unique_ptr<Pass> createDiffImplPass() {
  return std::make_unique<DiffImplPass>();
}
}  // namespace mlir::autodiff