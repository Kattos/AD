add_llvm_library(ADAutodiffPass
    Experimental.cpp
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
