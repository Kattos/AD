add_mlir_conversion_library(ADToTosaConversion
    ADToTosa.cpp

    DEPENDS
    ADConversionGen
)
