add_mlir_conversion_library(ADToCoreConversion
    ADToCore.cpp
    Helper.cpp

    DEPENDS
    ADConversionGen
    ADRules
)
