#include "mlir/Dialect/Math/IR/Math.h"

#include "Dialect/Grad/IR/GradInterface.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir {
namespace autodiff {

class LogPartial
    : public PartialInterface::ExternalModel<LogPartial, math::LogOp> {
 public:
  SmallVector<Value> partial(Operation* op, OpBuilder& builder) const {
    auto log = cast<math::LogOp>(op);
    auto type = log.getType();

    auto loc = builder.getUnknownLoc();
    auto attr = builder.getFloatAttr(type, 1.0);
    auto one = builder.create<arith::ConstantOp>(loc, attr);

    auto derivative = builder.create<arith::DivFOp>(loc, one, log.getOperand());
    return {derivative};
  }

  Value partialFor(Operation* op, OpBuilder& builder,
                   unsigned int index) const {
    if (index >= op->getNumOperands()) {
      return nullptr;
    }

    auto log = cast<math::LogOp>(op);
    auto type = log.getType();

    auto loc = builder.getUnknownLoc();
    auto attr = builder.getFloatAttr(type, 1.0);
    auto one = builder.create<arith::ConstantOp>(loc, attr);

    auto derivative = builder.create<arith::DivFOp>(loc, one, log.getOperand());
    return {derivative};
  }
};

void registerMathPartial(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, math::MathDialect*) {
    math::LogOp::attachInterface<LogPartial>(*context);
  });
}

}  // namespace autodiff
}  // namespace mlir
