#include "Conversion/GradToCore/GradToCore.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {
class GradMatMulToCore : public OpRewritePattern<grad::MatMulOp> {
  using OpRewritePattern<grad::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(grad::MatMulOp matmul,
                                PatternRewriter& rewriter) const override {
    auto dout = matmul.getDout();
    auto lhs = matmul.getLhs();
    auto rhs = matmul.getRhs();

    auto loc = rewriter.getUnknownLoc();
    auto permsType = RankedTensorType::get({3}, rewriter.getI32Type());
    auto permsValue = DenseIntElementsAttr::get(permsType, {0, 2, 1});
    auto perms = rewriter.create<tosa::ConstOp>(loc, permsType, permsValue);

    auto newType = [](Value v) -> RankedTensorType {
      auto oldType = v.getType().dyn_cast<RankedTensorType>();
      auto oldShape = oldType.getShape();
      auto newShape = {oldShape[0], oldShape[2], oldShape[1]};
      return RankedTensorType::get(newShape, oldType.getElementType());
    };

    auto transLhs =
        rewriter.create<tosa::TransposeOp>(loc, newType(lhs), lhs, perms);
    auto transRhs =
        rewriter.create<tosa::TransposeOp>(loc, newType(rhs), rhs, perms);

    auto dlhs =
        rewriter.create<tosa::MatMulOp>(loc, lhs.getType(), dout, transRhs);
    auto drhs =
        rewriter.create<tosa::MatMulOp>(loc, rhs.getType(), transLhs, dout);

    rewriter.replaceOp(matmul, {dlhs, drhs});
    return success();
  }
};

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
