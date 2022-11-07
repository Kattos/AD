set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.hpp.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(ADConversionPassIncGen)
