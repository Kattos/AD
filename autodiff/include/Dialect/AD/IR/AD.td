#ifndef AD_OPS
#define AD_OPS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def AD_Dialect : Dialect {
    let name = "ad";
    let cppNamespace = "::mlir::autodiff::ad";
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class AD_Op<string mnemonic, list<Trait> traits = []> : Op<AD_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Defines the operations of the dialect
//===----------------------------------------------------------------------===//

def AD_OneslikeOp : AD_Op<"oneslike"> {
    let summary = "Creates a tensor of all ones that has the same shape as the input";

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];
}

def AD_ZeroslikeOp : AD_Op<"zeroslike"> {
    let summary = "Creates a tensor of all zeros that has the same shape as the input";

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];
}

def AD_BackOp : AD_Op<"back"> {
    let summary = "Records the output of the computation graph";

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

#endif // AD_OPS
