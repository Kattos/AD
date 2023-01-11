#include "Rule/Rules.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir::autodiff {

using ArithmeticAddIRule = BinaryOpRule<arith::AddIOp>;
using ArithmeticAddFRule = BinaryOpRule<arith::AddFOp>;
using ArithmeticSubIRule = BinaryOpRule<arith::SubIOp>;
using ArithmeticSubFRule = BinaryOpRule<arith::SubFOp>;
using ArithmeticMulIRule = BinaryOpRule<arith::MulIOp>;
using ArithmeticMulFRule = BinaryOpRule<arith::MulFOp>;
using ArithmeticDivFRule = BinaryOpRule<arith::DivFOp>;
using ArithmeticMaxFRule = BinaryOpRule<arith::MaxFOp>;
using ArithmeticMaxSIRule = BinaryOpRule<arith::MaxSIOp>;
using ArithmeticMaxUIRule = BinaryOpRule<arith::MaxUIOp>;
using ArithmeticMinFRule = BinaryOpRule<arith::MinFOp>;
using ArithmeticMinSIRule = BinaryOpRule<arith::MinSIOp>;
using ArithmeticMinUIRule = BinaryOpRule<arith::MinUIOp>;
using ArithmeticNegFRule = UnaryOpRule<arith::NegFOp>;

ValueRange getArithGradients(Operation* op, Value grad);
Value getArithGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff
