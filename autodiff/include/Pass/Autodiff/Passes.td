#ifndef AUTODIFF_PASSES
#define AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

def DiffDeclPass : Pass<"ad-diff-decl"> {
    let summary = "Declare diff version of non-diff functions";
    let constructor = "mlir::autodiff::createDiffDeclPass()";
}

def DiffImplPass : Pass<"ad-diff-impl"> {
    let summary = "Implement diff version of non-diff functions";
    let constructor = "mlir::autodiff::createDiffImplPass()";
}

def DiffPass : Pass<"ad-diff"> {
    let summary = "Generate diff version of all functions";
    let constructor = "mlir::autodiff::createDiffPass()";
}

#endif // AUTODIFF_PASSES
