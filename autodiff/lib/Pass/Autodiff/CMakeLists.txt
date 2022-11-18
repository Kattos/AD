add_llvm_library(ADAutodiffPass
    Naive.cpp
    Experimental.cpp
    Grad.cpp
    Pipeline.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
    ADRules
)
