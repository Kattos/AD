#ifndef AUTODIFF_PASSES
#define AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

def PreprocessPass : Pass<"ad-preprocess"> {
    let summary = "Generate diff version of all functions but not implement them";
    let constructor = "mlir::autodiff::createADPreprocessPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect"];
}

def NaivePass : Pass<"ad-naive"> {
    let summary = "Apply autodiff on preprocessed functions naively";
    let constructor = "mlir::autodiff::createADNaivePass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect"];
}

def ExperimentalPass : Pass<"ad-experimental"> {
    let summary = "Used for testing purposes only";
    let constructor = "mlir::autodiff::createADExperimentalPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect"];
}

#endif // AUTODIFF_PASSES
