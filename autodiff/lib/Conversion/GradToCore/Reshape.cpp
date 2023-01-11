#include "Conversion/GradToCore/GradToCore.hpp"

namespace mlir {
namespace autodiff {
namespace grad {
namespace core {

Value dReshape(PatternRewriter& rewriter, Value output) {
  auto reshape = output.getDefiningOp<grad::ReshapeOp>();
  if (!reshape) {
    return nullptr;
  }

  auto loc = rewriter.getUnknownLoc();

  auto x = reshape.x();
  auto dout = reshape.dout();
  auto shape = x.getType().cast<ShapedType>().getShape();
  auto shapeAttr = rewriter.getI64ArrayAttr(shape);

  auto dx = rewriter.create<tosa::ReshapeOp>(loc, x.getType(), dout, shapeAttr);
  return dx;
}

}  // namespace core
}  // namespace grad
}  // namespace autodiff
}  // namespace mlir
