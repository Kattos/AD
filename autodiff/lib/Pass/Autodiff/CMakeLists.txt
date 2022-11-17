add_llvm_library(ADAutodiffPass
    Naive.cpp
    Experimental.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
    ADRules
)
