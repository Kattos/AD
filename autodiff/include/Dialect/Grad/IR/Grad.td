#ifndef GRAD_OPS
#define GRAD_OPS

include "mlir/IR/OpBase.td"

def Grad_Dialect : Dialect {
    let name = "grad";
    let cppNamespace = "::mlir::autodiff::grad";
}

class Grad_Op<string mnemonic, list<Trait> traits = []> : Op<Grad_Dialect, mnemonic, traits>;

def Grad_AbsOp : Grad_Op<"abs"> {
    let arguments = (ins AnyType:$x, AnyType:$dout);
    let results = (outs AnyType:$dx);
}

#endif // GRAD_OPS
