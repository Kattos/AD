add_mlir_dialect_library(ADDialect
    AD.cpp ADDialect.cpp

    DEPENDS
    MLIRADIncGen
)