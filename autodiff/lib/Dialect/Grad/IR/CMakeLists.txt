add_mlir_dialect_library(GradDialect
    Grad.cpp
    GradDialect.cpp
    GradInterface.cpp

    DEPENDS
    MLIRGradIncGen
    MLIRGradInterfaceIncGen
)
