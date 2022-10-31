#ifndef ADTOSA_OPS
#define ADTOSA_OPS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def ADTosa_Dialect : Dialect {
    let name = "ad_tosa";
    let cppNamespace = "::mlir::autodiff::ad_tosa";
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class ADTosa_Op<string mnemonic, list<Trait> traits = []> : Op<ADTosa_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Defines the operations of the dialect
//===----------------------------------------------------------------------===//

def ADTosa_OneslikeOp : ADTosa_Op<"oneslike"> {
    let summary = "Creates a tensor of all ones that has the same shape as the input";

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];
}

def ADTosa_ZeroslikeOp : ADTosa_Op<"zeroslike"> {
    let summary = "Creates a tensor of all zeros that has the same shape as the input";

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];
}

def ADTosa_GradLogOp : ADTosa_Op<"grad_log"> {
    let summary = "Grad version for tosa.log";

    let arguments = (ins AnyTensor:$grad, AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input)>,
        OpBuilder<(ins "Value":$grad, "Value":$input)>
    ];
}

#endif // ADTOSA_OPS
