#ifndef AD_CONVERSION_GRADTOCORE
#define AD_CONVERSION_GRADTOCORE

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "Dialect/AD/IR/AD.td"
include "Dialect/Grad/IR/Grad.td"

def ADD : NativeCodeCall<"add($_builder, $0, $1)">;
def MUL : NativeCodeCall<"mul($_builder, $0, $1)">;
def NEGATE : NativeCodeCall<"negate($_builder, $0)">;
def EXP : NativeCodeCall<"exp($_builder, $0)">;
def RECIPROCAL : NativeCodeCall<"reciprocal($_builder, $0)">;

// alias to simplify coding
def AT : AnyTypeOf<[AnyTensor]>;
def IT : AnyTypeOf<[I1Tensor, I8Tensor, I16Tensor, I32Tensor, I64Tensor]>;
def FT : AnyTypeOf<[F16Tensor, F32Tensor, F64Tensor]>;

def AddToCore : Pattern<
    (Grad_AddOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (AD_ReduceOp (MUL (AD_OneslikeOp (AD_BroadcastOp $lhs, $dout)), $dout), $lhs),
        (AD_ReduceOp (MUL (AD_OneslikeOp (AD_BroadcastOp $rhs, $dout)), $dout), $rhs),
    ]
>;

def SubToCore : Pattern<
    (Grad_SubOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (AD_ReduceOp (MUL (AD_OneslikeOp (AD_BroadcastOp $lhs, $dout)), $dout), $lhs),
        (AD_ReduceOp (MUL (NEGATE (AD_OneslikeOp (AD_BroadcastOp $rhs, $dout))), $dout), $rhs)
    ]
>;

def MulToCore : Pattern<
    (Grad_MulOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (AD_ReduceOp (MUL (AD_BroadcastOp $rhs, $dout), $dout), $lhs),
        (AD_ReduceOp (MUL (AD_BroadcastOp $lhs, $dout), $dout), $rhs)
    ]
>;

def ExpToCore : Pat<
    (Grad_ExpOp AT:$x, AT:$dout),
    (MUL (EXP $x), $dout)
>;

def LogToCore : Pat<
    (Grad_LogOp AT:$x, AT:$dout),
    (MUL (RECIPROCAL $x), $dout)
>;

def RsqrtToCore : Pat<
    (Grad_RsqrtOp AT:$x, AT:$dout),
    (MUL (NativeCodeCall<"drsqrt($_builder, $0)"> $x), $dout)
>;

// \derivative{tanh(x)}{x} = \frac{4}{(e^x + e^{-x})^2}
def TanhToCore : Pat<
    (Grad_TanhOp AT:$x, AT:$dout),
    (MUL (ADD (ADD:$half (RECIPROCAL:$quarter (MUL:$dominant (ADD:$sqrt (EXP $x), (EXP (NEGATE $x))), $sqrt)), $quarter), $half), $dout)
>;

def NegateToCore : Pat<
    (Grad_NegateOp AT:$x, AT:$dout),
    (MUL (NEGATE (AD_OneslikeOp $x)), $dout)
>;

def ReciprocalToCore : Pat<
    (Grad_ReciprocalOp AT:$x, AT:$dout),
    (MUL (NEGATE (RECIPROCAL (MUL $x, $x))), $dout)
>;

// \derivative{sigmoid(x)}{x} = \frac{e^{-x}}{(1 + e^{-x})^2}
def SigmoidToCore : Pat<
    (Grad_SigmoidOp AT:$x, AT:$dout),
    (MUL (MUL (RECIPROCAL (MUL (ADD:$sqrt (EXP:$ex (NEGATE $x)), (AD_OneslikeOp $ex)), $sqrt)), $ex), $dout)
>;

def AbsToCore : Pat<
    (Grad_AbsOp AT:$x, AT:$dout),
    (MUL (NativeCodeCall<"dabs($_builder, $0)"> $x), $dout)
>;

def MaximumToCore : Pattern<
    (Grad_MaximumOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (AD_ReduceOp (MUL (NativeCodeCall<"dGreaterEqual($_builder, $0, $1)"> (AD_BroadcastOp $lhs, $dout), (AD_BroadcastOp $rhs, $dout)), $dout), $lhs),
        (AD_ReduceOp (MUL (NativeCodeCall<"dGreaterEqual($_builder, $0, $1)"> (AD_BroadcastOp $rhs, $dout), (AD_BroadcastOp $lhs, $dout)), $dout), $rhs),
    ]
>;

def MinimumToCore : Pattern<
    (Grad_MinimumOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (AD_ReduceOp (MUL (NativeCodeCall<"dGreaterEqual($_builder, $1, $0)"> (AD_BroadcastOp $lhs, $dout), (AD_BroadcastOp $rhs, $dout)), $dout), $lhs),
        (AD_ReduceOp (MUL (NativeCodeCall<"dGreaterEqual($_builder, $1, $0)"> (AD_BroadcastOp $rhs, $dout), (AD_BroadcastOp $lhs, $dout)), $dout), $rhs),
    ]
>;

def IntClampToCore : Pat<
    (Grad_ClampOp IT:$x, IT:$dout, I64Attr:$min_int, I64Attr:$max_int, F32Attr:$min_fp, F32Attr:$max_fp),
    (MUL (Tosa_SelectOp (NativeCodeCall<"intClampHelper($_builder, $0, $1, $2)"> $x, $min_int, $max_int), (AD_OneslikeOp $x), (AD_ZeroslikeOp $x), (returnType $x)), $dout)
>;

def FloatClampToCore : Pat<
    (Grad_ClampOp FT:$x, FT:$dout, I64Attr:$min_int, I64Attr:$max_int, F32Attr:$min_fp, F32Attr:$max_fp),
    (MUL (Tosa_SelectOp (NativeCodeCall<"floatClampHelper($_builder, $0, $1, $2)"> $x, $min_fp, $max_fp), (AD_OneslikeOp $x), (AD_ZeroslikeOp $x), (returnType $x)), $dout)
>;

#endif // AD_CONVERSION_GRADTOCORE
