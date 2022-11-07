add_llvm_library(ADAutodiffPass
    Naive.cpp
    Preprocess.cpp
    OpHandler.cpp

    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
