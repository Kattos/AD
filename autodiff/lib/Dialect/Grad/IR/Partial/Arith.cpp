#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Dialect/Grad/IR/GradInterface.hpp"
#include "Util/Arith.hpp"
#include "Util/Utils.hpp"

namespace mlir {
namespace autodiff {

using namespace mlir::arith;
using namespace util::arith;

template <typename Impl, typename OpTy>
using Base = PartialInterface::ExternalModel<Impl, OpTy>;

class AddFPartial : public Base<AddFPartial, AddFOp> {
 public:
  Value partial(Operation* op, Value input, OpBuilder& builder) const {
    auto addf = cast<AddFOp>(op);
    auto lhs = addf.getLhs();
    auto rhs = addf.getRhs();

    if (input == lhs || input == rhs) {
      return constant(1.0, builder);
    }
    return constant(0.0, builder);
  }
};

class MulFPartial : public Base<MulFPartial, MulFOp> {
 public:
  Value partial(Operation* op, Value input, OpBuilder& builder) const {
    auto mulf = cast<MulFOp>(op);
    auto lhs = mulf.getLhs();
    auto rhs = mulf.getRhs();

    if (input == lhs) {
      return rhs;
    } else if (input == rhs) {
      return lhs;
    }

    return constant(0.0, builder);
  }
};

class SubFPartial : public Base<SubFPartial, SubFOp> {
 public:
  Value partial(Operation* op, Value input, OpBuilder& builder) const {
    auto subf = cast<SubFOp>(op);
    auto lhs = subf.getLhs();
    auto rhs = subf.getRhs();

    if (input == lhs) {
      return constant(1.0, builder);
    } else if (input == rhs) {
      return constant(-1.0, builder);
    }

    return constant(0.0, builder);
  }
};

class DivFPartial : public Base<DivFPartial, DivFOp> {
 public:
  SmallVector<Value> partial(Operation* op, OpBuilder& builder) const {
    return {};
  }

  Value partial(Operation* op, Value input, OpBuilder& builder) const {
    auto divf = cast<DivFOp>(op);
    auto lhs = divf.getLhs();
    auto rhs = divf.getRhs();

    if (input == lhs) {
      auto one = constant(1.0, builder);
      return div(one, rhs, builder);
    } else if (input == rhs) {
      auto neg = constant(-1.0, builder);
      auto sqr = mul(rhs, rhs, builder);
      auto negSqr = mul(neg, sqr, builder);
      return mul(lhs, negSqr, builder);
    }

    return constant(0.0, builder);
  }
};

void registerArithPartial(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, ArithDialect*) {
    AddFOp::attachInterface<AddFPartial>(*context);
    MulFOp::attachInterface<MulFPartial>(*context);
    SubFOp::attachInterface<SubFPartial>(*context);
    DivFOp::attachInterface<DivFPartial>(*context);
  });
}

}  // namespace autodiff
}  // namespace mlir
