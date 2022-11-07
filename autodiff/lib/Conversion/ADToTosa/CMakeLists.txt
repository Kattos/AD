add_mlir_dialect_library(ADConversionPass
    ADToTosa.cpp

    DEPENDS
    ADConversionPassIncGen
)