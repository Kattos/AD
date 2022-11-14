#include "../Rules.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::autodiff {

using ArithAddIRule = BinaryOpRule<arith::AddIOp>;
using ArithAddFRule = BinaryOpRule<arith::AddFOp>;
using ArithSubIRule = BinaryOpRule<arith::SubIOp>;
using ArithSubFRule = BinaryOpRule<arith::SubFOp>;
using ArithMulIRule = BinaryOpRule<arith::MulIOp>;
using ArithMulFRule = BinaryOpRule<arith::MulFOp>;
using ArithDivFRule = BinaryOpRule<arith::DivFOp>;

Value getArithGradient(Operation* op, Value grad, Value input);

}  // namespace mlir::autodiff
