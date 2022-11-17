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

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

def AD_ZeroslikeOp : AD_Op<"zeroslike", [Pure, SameOperandsAndResultType]> {
    let summary = "Creates a tensor of all zeros that has the same shape as the input";

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

// TODO: remove `from`, `to` and `return` if they are useless
def AD_FromOp : AD_Op<"from"> {
    let arguments = (ins AnyType:$input);
}

def AD_ToOp : AD_Op<"to", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

def AD_ReturnOp : AD_Op<"return"> {
    let arguments = (ins AnyType:$argument, AnyType:$grad);
}

def AD_PlaceholderOp : AD_Op<"placeholder", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

def AD_AbsOp : AD_Op<"abs", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType:$x);
    let results = (outs AnyType:$dx);
}

def AD_RsqrtOp : AD_Op<"rsqrt", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType:$x);
    let results = (outs AnyType:$dx);
}

def AD_GradAbsOp : AD_Op<"grad_abs"> {
    let arguments = (ins AnyType:$x, AnyType:$grad);
    let results = (outs AnyType:$dx);
}

#endif // AD_OPS
