add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Pass)
add_subdirectory(Rule)

add_llvm_library(InitAD 
    InitAD.cpp
    
    LINK_LIBS
    ADDialect
    ADAutodiffPass
    ADSimplifyPass
    ADToTosaConversion
    ADToCoreConversion
    ADRules
)
