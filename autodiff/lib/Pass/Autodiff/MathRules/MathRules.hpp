#include "../Rules.hpp"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::autodiff {

using MathAbsFRule = UnaryOpRule<math::AbsFOp>;
using MathAbsIRule = UnaryOpRule<math::AbsIOp>;

ValueRange getMathGradients(Operation* op, Value grad);
Value getMathGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff
