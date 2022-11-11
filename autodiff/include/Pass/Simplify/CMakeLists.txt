set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.hpp.inc -gen-pass-decls -name Simplify)
add_public_tablegen_target(ADSimplifyPassIncGen)
