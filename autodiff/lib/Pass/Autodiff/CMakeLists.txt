add_llvm_library(ADAutodiffPass
    Naive.cpp
    Experimental.cpp
    Grad.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
    ADRules
)
