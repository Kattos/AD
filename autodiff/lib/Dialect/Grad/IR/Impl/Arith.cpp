#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Dialect/Grad/IR/GradInterface.hpp"

namespace mlir {
namespace autodiff {

class AddFAdjoint
    : public AdjointInterface::ExternalModel<AddFAdjoint, arith::AddFOp> {
 public:
  ValueRange adjoint(Operation* op, OpBuilder& builder) const {
    static SmallVector<Value, 2> buffer;

    auto add = cast<arith::AddFOp>(op);
    auto type = add.getType();

    if (buffer.empty() || buffer[0].getType() != type) {
      auto loc = builder.getUnknownLoc();
      auto attr = builder.getFloatAttr(type, 1.0);
      auto one = builder.create<arith::ConstantOp>(loc, attr);
      buffer.clear();
      buffer.emplace_back(one);
      buffer.emplace_back(one);
    }

    return buffer;
  }
};

class MulFAdjoint
    : public AdjointInterface::ExternalModel<MulFAdjoint, arith::MulFOp> {};

void registerArithAdjoint(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, arith::ArithDialect*) {
    arith::AddFOp::attachInterface<AddFAdjoint>(*context);
  });
}

}  // namespace autodiff
}  // namespace mlir
