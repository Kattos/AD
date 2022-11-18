#ifndef AUTODIFF_PASSES
#define AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

// TODO: remove deprecated pass
def NaivePass : Pass<"ad-naive"> {
    let summary = "Apply autodiff on preprocessed functions naively";
    let constructor = "mlir::autodiff::createADNaivePass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::scf::SCFDialect", "mlir::tosa::TosaDialect"];
}

// TODO: rename this pass
def ExperimentalPass : Pass<"ad-experimental"> {
    let summary = "New features under development";
    let constructor = "mlir::autodiff::createADExperimentalPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::scf::SCFDialect", "mlir::tosa::TosaDialect"];
}

def GradPass : Pass<"ad-grad"> {
    let summary = "Generate gradients using `grad` ops";
    let constructor = "mlir::autodiff::createADGradPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::autodiff::grad::GradDialect"];
}

#endif // AUTODIFF_PASSES
