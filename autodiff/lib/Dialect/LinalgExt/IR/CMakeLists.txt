add_mlir_dialect_library(LinalgExtDialect
    LinalgExt.cpp

    DEPENDS
    MLIRLinalgExtIncGen
)