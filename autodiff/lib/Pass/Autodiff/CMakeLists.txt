add_llvm_library(ADAutodiffPass
    DiffDeclPass.cpp
    DiffImplPass.cpp
    TestReplacePass.cpp

    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADTosaDialect
)