add_mlir_dialect_library(ADTosaDialect
    ADTosa.cpp ADTosaDialect.cpp

    DEPENDS
    MLIRADTosaIncGen
)