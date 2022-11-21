#include "Rule/Utils.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::autodiff {

using CalFn =
    function_ref<Value(Operation *op, ValueRange args,
                       ArrayRef<Type> resultTypes, PatternRewriter &rewriter)>;

LogicalResult elementwiseMatchAndRewriteHelper(Operation *op,
                                               PatternRewriter &rewriter,
                                               CalFn calFn);

linalg::GenericOp buildGeneric(Operation *op, PatternRewriter &rewriter,
                               CalFn calFn);

linalg::GenericOp buildGeneric(Operation *op, ValueRange newOperands,
                               ValueRange newResults, PatternRewriter &rewriter,
                               CalFn calFn);

}  // namespace mlir::autodiff
