#ifndef AUTODIFF_PASSES
#define AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

def NaivePass : Pass<"ad-naive"> {
    let summary = "Apply autodiff on preprocessed functions naively";
    let constructor = "mlir::autodiff::createADNaivePass()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::scf::SCFDialect", "mlir::tosa::TosaDialect"];
}

#endif // AUTODIFF_PASSES
