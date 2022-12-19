#include "Conversion/GradToCore/GradToCore.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

/*

for i in range(lhs.shape[0]):
  for j in range(lhs.shape[1]):
    for k in range(lhs.shape[2]):
      for l in range(rhs.shape[2]):
        dx[i][j][k] += dout[i][j][l] * y[i][k][l]
        dy[i][k][l] += dout[i][j][l] * x[i][j][k]

*/

class GradMatMulToCore : public OpRewritePattern<grad::MatMulOp> {
  using OpRewritePattern<grad::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::MatMulOp matmul,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(matmul);
    auto loc = rewriter.getUnknownLoc();
    auto ctx = rewriter.getContext();

    constexpr auto DIM_COUNT = 4;
    constexpr auto SYM_COUNT = 0;
    constexpr auto RANKS = 3;

    SmallVector<AffineExpr, RANKS> exprs;
    auto getMap = [&](SmallVector<int> dims) -> AffineMap {
      exprs.clear();
      for (auto i : dims) {
        exprs.emplace_back(rewriter.getAffineDimExpr(i));
      }
      return AffineMap::get(DIM_COUNT, SYM_COUNT, exprs, ctx);
    };

    auto mapForDout = getMap({0, 1, 3});
    auto mapForL = getMap({0, 1, 2});
    auto mapForR = getMap({0, 2, 3});

    auto lhs = matmul.getLhs();
    auto rhs = matmul.getRhs();

    auto operandSegmentSizes = rewriter.getNamedAttr(
        "operand_segment_sizes", rewriter.getDenseI32ArrayAttr({0, 0}));

    SmallVector<Value, 0> dynamicSizes;
    auto dlhs = rewriter.create<bufferization::AllocTensorOp>(
        loc, lhs.getType(), dynamicSizes, operandSegmentSizes);
    auto drhs = rewriter.create<bufferization::AllocTensorOp>(
        loc, rhs.getType(), dynamicSizes, operandSegmentSizes);

    auto resultTensorTypes = matmul->getResultTypes();
    auto inputs = matmul->getOperands();
    auto outputs = ValueRange{dlhs, drhs};
    auto indexingMaps = {mapForL, mapForR, mapForDout, mapForL, mapForR};
    auto iteratorTypes =
        SmallVector<StringRef>(DIM_COUNT, getParallelIteratorTypeName());
    auto calculator = [](OpBuilder& builder, Location loc, ValueRange args) {
      auto lhs = args[0];
      auto rhs = args[1];
      auto dout = args[2];
      auto dlhs = args[3];
      auto drhs = args[4];

      auto leftDelta = builder.create<arith::MulFOp>(loc, rhs, dout);
      auto leftCurr = builder.create<arith::AddFOp>(loc, dlhs, leftDelta);

      auto rightDelta = builder.create<arith::MulFOp>(loc, lhs, dout);
      auto rightCurr = builder.create<arith::AddFOp>(loc, drhs, rightDelta);

      auto results = ValueRange{leftCurr.getResult(), rightCurr.getResult()};
      builder.create<linalg::YieldOp>(loc, results);
    };

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultTensorTypes, inputs, outputs, indexingMaps, iteratorTypes,
        calculator);

    rewriter.replaceOp(matmul, generic->getResults());

    return success();
  }
};

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
