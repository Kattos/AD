#include "Rule/Rules.hpp"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

using MathAbsRule = UnaryOpRule<math::AbsOp>;
using MathAbsRule = UnaryOpRule<math::AbsOp>;
using MathLogRule = UnaryOpRule<math::LogOp>;
using MathLog10Rule = UnaryOpRule<math::Log10Op>;
using MathLog2Rule = UnaryOpRule<math::Log2Op>;
using MathLog1pRule = UnaryOpRule<math::Log1pOp>;
using MathRsqrtRule = UnaryOpRule<math::RsqrtOp>;
using MathExpRule = UnaryOpRule<math::ExpOp>;
using MathTanhRule = UnaryOpRule<math::TanhOp>;

ValueRange getMathGradients(Operation* op, Value grad);
Value getMathGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff
