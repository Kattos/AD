#ifndef AUTODIFF_PASSES
#define AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

def NaivePass : Pass<"ad-naive"> {
    let summary = "Apply autodiff on preprocessed functions naively";
    let constructor = "mlir::autodiff::createADNaivePass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::scf::SCFDialect", "mlir::tosa::TosaDialect"];
}

def ExperimentalPass : Pass<"ad-experimental"> {
    let summary = "New features under development";
    let constructor = "mlir::autodiff::createADExperimentalPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::scf::SCFDialect"];
}

#endif // AUTODIFF_PASSES
