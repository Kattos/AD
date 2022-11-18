#include "Rule/Rules.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::autodiff {

using ArithAddIRule = BinaryOpRule<arith::AddIOp>;
using ArithAddFRule = BinaryOpRule<arith::AddFOp>;
using ArithSubIRule = BinaryOpRule<arith::SubIOp>;
using ArithSubFRule = BinaryOpRule<arith::SubFOp>;
using ArithMulIRule = BinaryOpRule<arith::MulIOp>;
using ArithMulFRule = BinaryOpRule<arith::MulFOp>;
using ArithDivFRule = BinaryOpRule<arith::DivFOp>;
using ArithMaxFRule = BinaryOpRule<arith::MaxFOp>;
using ArithMaxSIRule = BinaryOpRule<arith::MaxSIOp>;
using ArithMaxUIRule = BinaryOpRule<arith::MaxUIOp>;
using ArithMinFRule = BinaryOpRule<arith::MinFOp>;
using ArithMinSIRule = BinaryOpRule<arith::MinSIOp>;
using ArithMinUIRule = BinaryOpRule<arith::MinUIOp>;
using ArithNegFRule = UnaryOpRule<arith::NegFOp>;

ValueRange getArithGradients(Operation* op, Value grad);
Value getArithGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff
