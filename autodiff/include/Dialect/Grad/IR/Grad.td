#ifndef GRAD_OPS
#define GRAD_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect/Grad/IR/GradTypes.td"

def Grad_Dialect : Dialect {
    let name = "grad";
    let cppNamespace = "::mlir::autodiff::grad";
}

class Grad_Op<string mnemonic, list<Trait> traits = []> : Op<Grad_Dialect, mnemonic, traits>;

def SameInputAndDerivativeType : NativeOpTrait<"SameInputAndDerivativeType">;

class Grad_UnaryOp<string mnemonic, list<Trait> traits=[Pure]> : Grad_Op<mnemonic, traits> {
    let arguments = (ins AnyType: $x, AnyType: $dout);
    let results = (outs AnyType: $dx);
}

def Grad_AbstractUnaryOp : Grad_UnaryOp<"abstract_unary"> {
    let arguments = (ins AnyType: $x, AnyType: $dout, StrAttr: $op);
}

def Grad_AbsOp : Grad_UnaryOp<"abs", [SameOperandsAndResultType]>;

def Grad_ExpOp : Grad_UnaryOp<"exp", [SameOperandsAndResultType]>;

def Grad_LogOp : Grad_UnaryOp<"log", [SameOperandsAndResultType]>;

def Grad_RsqrtOp : Grad_UnaryOp<"rsqrt", [SameOperandsAndResultType]>;

def Grad_TanhOp :  Grad_UnaryOp<"tanh", [SameOperandsAndResultType]>;

def Grad_ClampOp : Grad_UnaryOp<"clamp", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType: $x, AnyType: $dout, I64Attr: $min_int, I64Attr: $max_int, F32Attr: $min_fp, F32Attr: $max_fp);
}

def Grad_NegateOp : Grad_UnaryOp<"negate", [SameOperandsAndResultType]>;

def Grad_ReciprocalOp : Grad_UnaryOp<"reciprocal", [SameOperandsAndResultType]>;

def Grad_SigmoidOp : Grad_UnaryOp<"sigmoid", [SameOperandsAndResultType]>;

def Grad_ReshapeOp : Grad_UnaryOp<"reshape"> {
    let arguments = (ins
        AnyType: $x,
        AnyType: $dout,
        I64ArrayAttr: $new_shape
    );
}

def Grad_ReduceSumOp : Grad_UnaryOp<"reduce_sum">;

class Grad_BinaryOp<string mnemonic, list<Trait> traits = [SameInputAndDerivativeType]> : Grad_Op<mnemonic, traits> {
    let arguments = (ins AnyType: $lhs, AnyType: $rhs, AnyType: $dout);
    let results = (outs AnyType: $dlhs, AnyType: $drhs);
}

def Grad_AbstractBinaryOp : Grad_BinaryOp<"abstract_binary"> {
    let arguments = (ins AnyType: $lhs, AnyType: $rhs, AnyType: $dout, StrAttr: $op);
}

def Grad_AddOp : Grad_BinaryOp<"add">;

def Grad_SubOp : Grad_BinaryOp<"sub">;

def Grad_MulOp : Grad_BinaryOp<"mul">;

def Grad_MaximumOp : Grad_BinaryOp<"maximum">;

def Grad_MinimumOp : Grad_BinaryOp<"minimum">;

def Grad_PowOp : Grad_BinaryOp<"pow">;

def Grad_AnyNumber : AnyTypeOf<[AnyInteger, AnyFloat]>;

def Grad_AvgPool2dOp : Grad_UnaryOp<"avg_pool2d"> {
    let arguments = (ins
        Grad_Tensor4D: $x,
        Grad_Tensor4D: $dout,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>: $kernel,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>: $stride,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>: $pad
    );

    let results = (outs 4DTensorOf<[Grad_AnyNumber]>: $dx);
}

// TODO: add `SameInputAndDerivativeType` trait
def Grad_Conv2DOp : Grad_Op<"conv2d"> {
    let arguments = (ins
        Grad_Tensor4D: $x,
        Grad_Tensor4D: $weight,
        Grad_Tensor1D: $bias,
        Grad_Tensor4D: $dout,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>: $dilation,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>: $stride,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>: $pad
    );

    let results = (outs 
        Grad_Tensor4D: $dx,
        Grad_Tensor4D: $dweight,
        Grad_Tensor1D: $dbias
    );
}

def Grad_MatMulOp : Grad_BinaryOp<"matmul"> {
    let arguments = (ins
        Grad_Tensor3D: $lhs,
        Grad_Tensor3D: $rhs,
        Grad_Tensor3D: $dout
    );

    let results = (outs
        Grad_Tensor3D: $dlhs,
        Grad_Tensor3D: $drhs
    );
}

def Grad_NablaOp : Grad_Op<"nabla", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins
        FlatSymbolRefAttr:$func,
        Variadic<AnyTensor>:$inputs
    );

    let results = (outs 
        Variadic<AnyTensor>:$outputs
    );
}

#endif // GRAD_OPS
