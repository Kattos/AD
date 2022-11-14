add_llvm_library(ADAutodiffPass
    ADUtils.cpp
    Naive.cpp
    Rules.cpp
    TosaRules.cpp
    MathRules.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
