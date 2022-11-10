add_llvm_library(ADAutodiffPass
    ADUtils.cpp
    Naive.cpp
    Rules.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
