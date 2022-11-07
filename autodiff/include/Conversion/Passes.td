#ifndef AD_CONVERSION_PASSES
#define AD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ADToTosa : Pass<"ad-to-tosa"> {
    let summary = "Convert AD ops to TOSA ops";
    let constructor = "mlir::autodiff::createADToTosa()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::tosa::TosaDialect", "mlir::bufferization::BufferizationDialect"];
}

#endif // AD_CONVERSION_PASSES