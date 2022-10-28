get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        InitAD
)

add_llvm_executable(autodiff-opt autodiff-opt.cpp)
target_link_libraries(autodiff-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(autodiff-opt)
