add_llvm_library(ADAutodiffPass
    GenGrad.cpp
    GradGeneric.cpp
    Pipeline.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
    ADRules
)
