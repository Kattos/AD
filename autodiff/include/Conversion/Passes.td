#ifndef AD_CONVERSION_PASSES
#define AD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ADToCore : Pass<"ad-to-core"> {
    let summary = "Convert AD ops to ops from mlir core dialects";
    let constructor = "mlir::autodiff::createADToCore()";

    let dependentDialects = ["mlir::autodiff::ad::ADDialect", "mlir::tensor::TensorDialect", "mlir::tosa::TosaDialect", "mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect", "mlir::bufferization::BufferizationDialect"];
}

def GradToCore : Pass<"grad-to-core"> {
    let summary = "Convert Grad ops to ops from mlir core dialects";
    let constructor = "mlir::autodiff::createGradToCore()";

    let dependentDialects = [
        "mlir::autodiff::ad::ADDialect", 
        "mlir::math::MathDialect", 
        "mlir::scf::SCFDialect", 
        "mlir::tensor::TensorDialect", 
        "mlir::linalg::LinalgDialect", 
        "mlir::tosa::TosaDialect", 
        "mlir::bufferization::BufferizationDialect"
    ];
}

def GradAbstractToConcrete : Pass<"grad-concrete"> {
    let summary = "Convert grad abstract ops to concrete ops";
    let constructor = "mlir::autodiff::createGradAbstractToConcrete()";

    let dependentDialects = ["mlir::autodiff::grad::GradDialect", "mlir::tosa::TosaDialect"];
}

def InitTensorToAllocTensor : Pass<"init-tensor-to-alloc-tensor"> {
    let summary = "Convert `linalgext.init_tensor` to `bufferization.alloc_tensor`";
    let constructor = "mlir::autodiff::createInitTensorToAllocTensor()";

    let dependentDialects = ["mlir::autodiff::linalgext::LinalgExtDialect", "mlir::bufferization::BufferizationDialect"];
}

def AllocTensorToInitTensor : Pass<"alloc-tensor-to-init-tensor"> {
    let summary = "Convert `bufferization.alloc_tensor` to `linalgext.init_tensor`";
    let constructor = "mlir::autodiff::createAllocTensorToInitTensor()";

    let dependentDialects = ["mlir::autodiff::linalgext::LinalgExtDialect", "mlir::bufferization::BufferizationDialect"];
}

#endif // AD_CONVERSION_PASSES
