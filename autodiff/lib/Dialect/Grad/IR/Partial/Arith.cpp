#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Dialect/Grad/IR/GradInterface.hpp"

namespace mlir {
namespace autodiff {

class AddFPartial
    : public PartialInterface::ExternalModel<AddFPartial, arith::AddFOp> {
 public:
  SmallVector<Value> partial(Operation* op, OpBuilder& builder) const {
    auto one = partialFor(op, builder, 0);
    return {one, one};
  }

  Value partialFor(Operation* op, OpBuilder& builder,
                   unsigned int index) const {
    if (index >= op->getNumOperands()) {
      return nullptr;
    }

    auto addf = cast<arith::AddFOp>(op);
    auto type = addf.getType();

    auto loc = builder.getUnknownLoc();
    auto attr = builder.getFloatAttr(type, 1.0);
    auto one = builder.create<arith::ConstantOp>(loc, attr);

    return one;
  }
};

class MulFPartial
    : public PartialInterface::ExternalModel<MulFPartial, arith::MulFOp> {
 public:
  SmallVector<Value> partial(Operation* op, OpBuilder& builder) const {
    auto mulf = cast<arith::MulFOp>(op);

    return {mulf.getRhs(), mulf.getLhs()};
  }

  Value partialFor(Operation* op, OpBuilder& builder,
                   unsigned int index) const {
    if (index >= op->getNumOperands()) {
      return nullptr;
    }
    auto mulf = cast<arith::MulFOp>(op);

    return index == 0 ? mulf.getRhs() : mulf.getLhs();
  }
};

class SubFPartial
    : public PartialInterface::ExternalModel<SubFPartial, arith::SubFOp> {
 public:
  SmallVector<Value> partial(Operation* op, OpBuilder& builder) const {
    auto subf = cast<arith::SubFOp>(op);
    auto type = subf.getType();

    auto loc = builder.getUnknownLoc();
    auto posAttr = builder.getFloatAttr(type, 1.0);
    auto pos = builder.create<arith::ConstantOp>(loc, posAttr);
    auto negAttr = builder.getFloatAttr(type, -1.0);
    auto neg = builder.create<arith::ConstantOp>(loc, negAttr);

    return {pos, neg};
  }

  Value partialFor(Operation* op, OpBuilder& builder,
                   unsigned int index) const {
    if (index >= op->getNumOperands()) {
      return nullptr;
    }

    auto addf = cast<arith::SubFOp>(op);
    auto type = addf.getType();

    auto loc = builder.getUnknownLoc();
    auto attr = builder.getFloatAttr(type, index == 0 ? 1.0 : -1.0);
    auto one = builder.create<arith::ConstantOp>(loc, attr);

    return one;
  }
};

void registerArithPartial(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, arith::ArithDialect*) {
    arith::AddFOp::attachInterface<AddFPartial>(*context);
    arith::MulFOp::attachInterface<MulFPartial>(*context);
    arith::SubFOp::attachInterface<SubFPartial>(*context);
  });
}

}  // namespace autodiff
}  // namespace mlir
