add_llvm_library(ADAutodiffPass
    ADUtils.cpp
    Naive.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
