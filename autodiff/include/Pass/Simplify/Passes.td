#ifndef SIMPLIFY_PASSES
#define SIMPLIFY_PASSES


include "mlir/Pass/PassBase.td"
include "Dialect/AD/IR/AD.td"

def SimplifyPass : Pass<"ad-simplify"> {
    let summary = "Remove redundant operations";
    let constructor = "mlir::autodiff::createADSimplifyPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect"];
}

#endif // SIMPLIFY_PASSES
