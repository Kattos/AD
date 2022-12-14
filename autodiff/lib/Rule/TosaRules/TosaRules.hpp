#include "Rule/Rules.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::autodiff {

using TosaLogRule = UnaryOpRule<tosa::LogOp>;
using TosaExpRule = UnaryOpRule<tosa::ExpOp>;
using TosaNegateRule = UnaryOpRule<tosa::NegateOp>;

using TosaAddRule = BinaryOpRule<tosa::AddOp>;
using TosaSubRule = BinaryOpRule<tosa::SubOp>;
using TosaMulRule = BinaryOpRule<tosa::MulOp>;

ValueRange getTosaGradients(Operation* op, Value grad);
Value getTosaGradient(Operation* op, Value grad, Value input);
}  // namespace mlir::autodiff
