add_llvm_library(ADAutodiffPass
    ADUtils.cpp
    Naive.cpp
    Preprocess.cpp
    Experimental.cpp

    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
