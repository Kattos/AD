add_llvm_library(ADConversion
    ADToCore/ADToCore.cpp
    GradToCore/GradToCore.cpp
    GradAbstractToConcrete/GradAbstractToConcrete.cpp

    DEPENDS
    ADConversionGen
    ADRules
    ADToCoreGen
    GradAbstractToConcreteGen
    GradToCoreGen
)
