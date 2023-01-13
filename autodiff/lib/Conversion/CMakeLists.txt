add_llvm_library(ADConversion
    ADToCore/ADToCore.cpp
    GradToCore/GradToCore.cpp
    GradToCore/Utils.cpp
    GradAbstractToConcrete/GradAbstractToConcrete.cpp
    LinalgExtConversion/LinalgExtConversion.cpp
    NablaLowering/NablaLowering.cpp

    DEPENDS
    ADConversionGen
    ADRules
    ADToCoreGen
    GradAbstractToConcreteGen
    GradToCoreGen
)
