#include "Rule/Utils.hpp"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::autodiff {

using CalFn =
    function_ref<Value(Operation *op, ValueRange args,
                       ArrayRef<Type> resultTypes, PatternRewriter &rewriter)>;

LogicalResult elementwiseMatchAndRewriteHelper(Operation *op,
                                               PatternRewriter &rewriter,
                                               CalFn loopFn);

}  // namespace mlir::autodiff
