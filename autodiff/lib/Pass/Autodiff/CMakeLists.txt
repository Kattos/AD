add_llvm_library(ADAutodiffPass
    DiffDeclPass.cpp
    DiffImplPass.cpp
    TestReplacePass.cpp

    DEPENDS
    MLIRIR
    MLIRAutodiffPassIncGen
    MLIRTransforms
    ADTosaDialect
)