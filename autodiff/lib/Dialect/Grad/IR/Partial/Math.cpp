#include "mlir/Dialect/Math/IR/Math.h"

#include "Dialect/Grad/IR/GradInterface.hpp"
#include "Util/Utils.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace autodiff {

using namespace mlir::math;
using namespace util::arith;

template <typename Impl, typename OpTy>
using Base = PartialInterface::ExternalModel<Impl, OpTy>;

class LogPartial : public Base<LogPartial, LogOp> {
 public:
  Value partial(Operation* op, Value input, OpBuilder& builder) const {
    auto index = util::others::indexOfOperand(op, input);
    if (index.has_value()) {
      auto one = constant(1.0, builder);
      return div(one, input, builder);
    }

    return constant(0.0, builder);
  }
};

class RsqrtPartial : public Base<RsqrtPartial, RsqrtOp> {
 public:
  Value partial(Operation* op, Value input, OpBuilder& builder) const {
    auto rsqrt = cast<RsqrtOp>(op);
    if (input == rsqrt.getOperand()) {
      auto coefficient = constant(-0.5, builder);
      auto exponent = constant(-1.5, builder);
      auto pow =
          builder.create<PowFOp>(builder.getUnknownLoc(), input, exponent);
      return mul(coefficient, pow, builder);
    }
    return constant(0.0, builder);
  }
};

void registerMathPartial(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, MathDialect*) {
    LogOp::attachInterface<LogPartial>(*context);
    RsqrtOp::attachInterface<RsqrtPartial>(*context);
  });
}

}  // namespace autodiff
}  // namespace mlir
