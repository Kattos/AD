add_llvm_library(ADAutodiffPass
    DiffDeclPass.cpp
    DiffImplPass.cpp
    DiffPass.cpp

    DEPENDS
    MLIRIR
    MLIRTransforms
    ADAutodiffPassIncGen
    ADDialect
)
