#ifndef NABLA_OPS
#define NABLA_OPS

include "mlir/IR/OpBase.td"

def Nabla_Dialect : Dialect {
    let name = "nabla";
    let cppNamespace = "::mlir::autodiff::nabla";
}

class Nabla_Op<string mnemonic, list<Trait> traits = []> : Op<Nabla_Dialect, mnemonic, traits>;

def Nabla_GradientOp : Nabla_Op<"gradient"> {
    let arguments = (ins
        AnyFloat: $target,
        AnyFloat: $source
    );

    let results = (outs
        AnyFloat: $gradient
    );
}

#endif // NABLA_OPS
