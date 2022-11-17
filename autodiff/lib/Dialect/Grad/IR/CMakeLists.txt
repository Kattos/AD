add_mlir_dialect_library(GradDialect
    Grad.cpp
    GradDialect.cpp

    DEPENDS
    MLIRGradIncGen
)
