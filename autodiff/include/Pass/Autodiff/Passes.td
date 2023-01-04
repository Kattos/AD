#ifndef AUTODIFF_PASSES
#define AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

def GenGradPass : Pass<"ad-gen-grad"> {
    let summary = "Generate gradients using `grad` ops";
    let constructor = "mlir::autodiff::createADGenGradPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::autodiff::grad::GradDialect"];
}

def GradGenericPass : Pass<"ad-grad-generic"> {
    let summary = "Generate gradients for `linalg.geneirc` ops";
    let constructor = "mlir::autodiff::createADGradGenericPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::autodiff::grad::GradDialect"];
}

def ExperimentalPass : Pass<"experimental"> {
    let summary = "Pass for experimental purposes";
    let constructor = "mlir::autodiff::createExperimentalPass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::autodiff::grad::GradDialect"];
}

#endif // AUTODIFF_PASSES
