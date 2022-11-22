set(LLVM_TARGET_DEFINITIONS ADToCore.td)
mlir_tablegen(ADToCore.hpp.inc -gen-rewriters -name ADToCore)
add_public_tablegen_target(ADToCoreGen)
