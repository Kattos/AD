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
    let summary = "Create a tensor of all ones that has the same shape as the input";

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

def AD_ZeroslikeOp : AD_Op<"zeroslike", [Pure, SameOperandsAndResultType]> {
    let summary = "Create a tensor of all zeros that has the same shape as the input";

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

def ResultsAreScalarTensorLike : NativeOpTrait<"ResultsAreScalarTensorLike">;

def AD_ScalarTensorOp : AD_Op<"scalar_tensor", [ResultsAreScalarTensorLike]> {
    let summary = "Create a tensor of input scalar";

    let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$input);
    let results = (outs AnyTensor:$tensor);

    let builders = [
        OpBuilder<(ins "Value":$input)>,
        OpBuilder<(ins "Type":$type, "int64_t":$literal)>,
        OpBuilder<(ins "Type":$type, "double":$literal)>
    ];
}

def AD_ToTensorOp : AD_Op<"to_tensor"> {
    let summary = "Create a tensor of any input";

    let arguments = (ins AnyType:$input);
    let results = (outs AnyTensor:$tensor);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];
}

def SameToAndResultType : NativeOpTrait<"SameToAndResultType">;

def AD_BroadcastOp : AD_Op<"broadcast", [SameToAndResultType, SameOperandsElementType]> {
    let arguments = (ins AnyTensor:$from, AnyTensor:$to);
    let results = (outs AnyType:$tensor);

    let builders = [
        OpBuilder<(ins "Value":$from, "Value":$to)>
    ];
}

def AD_ReduceOp : AD_Op<"reduce", [SameToAndResultType, SameOperandsElementType]> {
    let arguments = (ins AnyTensor:$from, AnyTensor:$to);
    let results = (outs AnyType:$tensor);

    let builders = [
        OpBuilder<(ins "Value":$from, "Value":$to)>
    ];
}

def AD_ReturnOp : AD_Op<"return"> {
    let arguments = (ins AnyType:$argument, AnyType:$grad);
}

def AD_PlaceholderOp : AD_Op<"placeholder", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
}

#endif // AD_OPS
