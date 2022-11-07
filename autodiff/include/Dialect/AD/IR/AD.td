#ifndef AD_OPS
#define AD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


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

def AD_OneslikeOp : AD_Op<"oneslike", [Pure, SameOperandsAndResultType]> {
    let summary = "Creates a tensor of all ones that has the same shape as the input";

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def AD_ZeroslikeOp : AD_Op<"zeroslike", [Pure, SameOperandsAndResultType]> {
    let summary = "Creates a tensor of all zeros that has the same shape as the input";

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def AD_FromOp : AD_Op<"from"> {
    let arguments = (ins AnyTensor:$input);
}

// FIXME: seems like `to` op is useless
def AD_ToOp : AD_Op<"to", [SameOperandsAndResultType]> {
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def AD_ReturnOp : AD_Op<"return"> {
    let arguments = (ins AnyTensor:$argument, AnyTensor:$grad);
}

#endif // AD_OPS
