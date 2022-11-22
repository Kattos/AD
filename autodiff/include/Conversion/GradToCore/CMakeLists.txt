set(LLVM_TARGET_DEFINITIONS GradToCore.td)
mlir_tablegen(GradToCore.hpp.inc -gen-rewriters -name GradToCore)
add_public_tablegen_target(GradToCoreGen)
