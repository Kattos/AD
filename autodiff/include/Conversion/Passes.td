#ifndef AD_CONVERSION_PASSES
#define AD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

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

def GradAbstractToConcrete : Pass<"grad-concrete"> {
    let summary = "Convert grad abstract ops to concrete ops";
    let constructor = "mlir::autodiff::createGradAbstractToConcrete()";

    let dependentDialects = ["mlir::autodiff::grad::GradDialect", "mlir::tosa::TosaDialect"];
}

#endif // AD_CONVERSION_PASSES
