#ifndef GRAD_INTERFACE
#define GRAD_INTERFACE

include "mlir/IR/OpBase.td"

def AdjointInterface : OpInterface<"AdjointInterface"> {
    let description = [{
       Generate partial derivative for each input of the operation 
    }];
    
    let cppNamespace = "::mlir::autodiff";

    let methods = [
        InterfaceMethod<
            /*desc*/ "Generate partial derivative for each input of the operation",
            /*retTy*/ "::mlir::ValueRange",
            /*methodName*/ "adjoint",
            /*args*/ (ins "::mlir::OpBuilder&":$builder)
        >
    ];
}

#endif
