#ifndef GRAD_INTERFACE
#define GRAD_INTERFACE

include "mlir/IR/OpBase.td"

def PartialInterface : OpInterface<"PartialInterface"> {
    let description = [{
       Generate partial derivative for each input of the operation 
    }];
    
    let cppNamespace = "::mlir::autodiff";

    let methods = [
        InterfaceMethod<
            /*desc*/ "Generate partial derivative for each input of the operation",
            /*retTy*/ "llvm::SmallVector<::mlir::Value>",
            /*methodName*/ "partial",
            /*args*/ (ins "::mlir::OpBuilder&":$builder)
        >,
        InterfaceMethod<
            "Generate partial derivative for specified input",
            "::mlir::Value",
            "partialFor",
            (ins "::mlir::OpBuilder&":$builder, "unsigned int":$index)
        >,
    ];
}

#endif
