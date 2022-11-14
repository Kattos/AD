add_llvm_library(ADAutodiffPass
    ADUtils.cpp
    Naive.cpp
    Rules.cpp
    TosaRules/TosaRules.cpp
    MathRules/MathRules.cpp
    ArithRules/ArithRules.cpp
    
    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
