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

def AD_FromOp : AD_Op<"from"> {
    let arguments = (ins AnyType:$input);
}


// FIXME: seems like `to` op is useless
def AD_ToOp : AD_Op<"to"> {
    let arguments = (ins AnyType:$input);
}

def AD_ReturnOp : AD_Op<"return"> {
    let arguments = (ins AnyType:$argument, AnyType:$grad);
}

// TODO: remove unused operations
def AD_BackfromOp : AD_Op<"backfrom"> {
    let summary = "Records the outputs of the computation graph";

    let arguments = (ins AnyType:$input);
    // let results = (outs AnyType:$output);
}

def AD_BacktoOp : AD_Op<"backto"> {
    let summary = "Records the inputs of the computation graph";

    let arguments = (ins AnyType:$input);
    // let results = (outs AnyType:$output);
}

#endif // AD_OPS
