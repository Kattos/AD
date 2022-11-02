add_llvm_library(ADAutodiffPass
    Naive.cpp
    Preprocess.cpp

    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
