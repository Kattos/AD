#include "../Rules.hpp"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

using MathAbsFRule = UnaryOpRule<math::AbsFOp>;
using MathAbsIRule = UnaryOpRule<math::AbsIOp>;
using MathLogRule = UnaryOpRule<math::LogOp>;
using MathLog10Rule = UnaryOpRule<math::Log10Op>;
using MathLog2Rule = UnaryOpRule<math::Log2Op>;
using MathLog1pRule = UnaryOpRule<math::Log1pOp>;

ValueRange getMathGradients(Operation* op, Value grad);
Value getMathGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff
