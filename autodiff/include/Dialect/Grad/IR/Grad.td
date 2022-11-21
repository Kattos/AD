#ifndef GRAD_OPS
#define GRAD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Grad_Dialect : Dialect {
    let name = "grad";
    let cppNamespace = "::mlir::autodiff::grad";
}

class Grad_Op<string mnemonic, list<Trait> traits = []> : Op<Grad_Dialect, mnemonic, traits>;

def SameInputAndDerivativeType : NativeOpTrait<"SameInputAndDerivativeType">;

class Grad_UnaryOp<string mnemonic, list<Trait> traits=[Pure, SameOperandsAndResultType]> : Grad_Op<mnemonic, traits> {
    let arguments = (ins AnyType:$x, AnyType:$dout);
    let results = (outs AnyType:$dx);
}

def Grad_AbstractUnaryOp : Grad_UnaryOp<"abstract_unary"> {
    let arguments = (ins AnyType:$x, AnyType:$dout, StrAttr:$op);
}

class Grad_BinaryOp<string mnemonic, list<Trait> traits = [SameInputAndDerivativeType]> : Grad_Op<mnemonic, traits> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$dout);
    let results = (outs AnyType:$dlhs, AnyType:$drhs);
}

def Grad_AbstractBinaryOp : Grad_BinaryOp<"abstract_binary"> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$dout, StrAttr:$op);
}

class Grad_BinaryLhsOp<string mnemonic, list<Trait> traits = []> : Grad_Op<mnemonic, traits> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$dout);
    let results = (outs AnyType:$dlhs);
}

class Grad_BinaryRhsOp<string mnemonic, list<Trait> traits = []> : Grad_Op<mnemonic, traits> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$dout);
    let results = (outs AnyType:$drhs);
}

def Grad_AbsOp : Grad_UnaryOp<"abs">;

def Grad_ExpOp : Grad_UnaryOp<"exp">;

def Grad_LogOp : Grad_UnaryOp<"log">;

def Grad_RsqrtOp : Grad_UnaryOp<"rsqrt">;

def Grad_TanhOp :  Grad_UnaryOp<"tanh">;

def Grad_ClampOp : Grad_UnaryOp<"clamp"> {
    let arguments = (ins AnyType:$x, AnyType:$dout, I64Attr:$min_int, I64Attr:$max_int, F32Attr:$min_fp, F32Attr:$max_fp);
}

def Grad_NegateOp : Grad_UnaryOp<"negate">;

def Grad_ReciprocalOp : Grad_UnaryOp<"reciprocal">;

def Grad_SigmoidOp : Grad_UnaryOp<"sigmoid">;

def Grad_AddOp : Grad_BinaryOp<"add">;

def Grad_AddLhsOp : Grad_BinaryLhsOp<"add_lhs">;

def Grad_AddRhsOp : Grad_BinaryRhsOp<"add_rhs">;

#endif // GRAD_OPS
