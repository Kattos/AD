#ifndef AD_CONVERSION_PASSES
#define AD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// TODO: remove deprecated pass
def ADToTosa : Pass<"ad-to-tosa"> {
    let summary = "Convert AD ops to TOSA ops";
    let constructor = "mlir::autodiff::createADToTosa()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::tosa::TosaDialect", "mlir::bufferization::BufferizationDialect"];
}

def ADToCore : Pass<"ad-to-core"> {
    let summary = "Convert AD ops to ops from mlir core dialects";
    let constructor = "mlir::autodiff::createADToCore()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::tensor::TensorDialect", "mlir::tosa::TosaDialect", "mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect"];
}

def GradToCore : Pass<"grad-to-core"> {
    let summary = "Convert Grad ops to ops from mlir core dialects";
    let constructor = "mlir::autodiff::createGradToCore()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::math::MathDialect", "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect", "mlir::tosa::TosaDialect"];
}

#endif // AD_CONVERSION_PASSES
