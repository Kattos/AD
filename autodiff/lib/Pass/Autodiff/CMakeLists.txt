add_llvm_library(ADAutodiffPass
    Naive.cpp
    Experimental.cpp
    GenGrad.cpp
    Pipeline.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
    ADRules
)
