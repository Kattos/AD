add_llvm_library(ADConversion
    ADToCore/ADToCore.cpp
    ADToTosa/ADToTosa.cpp
    GradToCore/GradToCore.cpp

    DEPENDS
    ADConversionGen
    ADRules
)
