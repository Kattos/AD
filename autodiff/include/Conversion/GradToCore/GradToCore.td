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

def ONESLIKE : NativeCodeCall<"oneslike($_builder, $0)">;
def BROADCAST : NativeCodeCall<"broadcast($_builder, $0, $1)">;
def REDUCE : NativeCodeCall<"reduce($_builder, $0, $1)">;

// alias to simplify coding
def AT : AnyTypeOf<[AnyTensor]>;
def IT : AnyTypeOf<[I1Tensor, I8Tensor, I16Tensor, I32Tensor, I64Tensor]>;
def FT : AnyTypeOf<[F16Tensor, F32Tensor, F64Tensor]>;

def AddToCore : Pattern<
    (Grad_AddOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (REDUCE (MUL (ONESLIKE (BROADCAST $lhs, $dout)), $dout), $lhs),
        (REDUCE (MUL (ONESLIKE (BROADCAST $rhs, $dout)), $dout), $rhs),
    ]
>;

def SubToCore : Pattern<
    (Grad_SubOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (REDUCE (MUL (ONESLIKE (BROADCAST $lhs, $dout)), $dout), $lhs),
        (REDUCE (MUL (NEGATE (ONESLIKE (BROADCAST $rhs, $dout))), $dout), $rhs)
    ]
>;

def MulToCore : Pattern<
    (Grad_MulOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (REDUCE (MUL (BROADCAST $rhs, $dout), $dout), $lhs),
        (REDUCE (MUL (BROADCAST $lhs, $dout), $dout), $rhs)
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
    (MUL (NEGATE (ONESLIKE $x)), $dout)
>;

def ReciprocalToCore : Pat<
    (Grad_ReciprocalOp AT:$x, AT:$dout),
    (MUL (NEGATE (RECIPROCAL (MUL $x, $x))), $dout)
>;

// \derivative{sigmoid(x)}{x} = \frac{e^{-x}}{(1 + e^{-x})^2}
def SigmoidToCore : Pat<
    (Grad_SigmoidOp AT:$x, AT:$dout),
    (MUL (MUL (RECIPROCAL (MUL (ADD:$sqrt (EXP:$ex (NEGATE $x)), (ONESLIKE $ex)), $sqrt)), $ex), $dout)
>;

def AbsToCore : Pat<
    (Grad_AbsOp AT:$x, AT:$dout),
    (MUL (NativeCodeCall<"dabs($_builder, $0)"> $x), $dout)
>;

def MaximumToCore : Pattern<
    (Grad_MaximumOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (REDUCE (MUL (NativeCodeCall<"dGreaterEqual($_builder, $0, $1)"> (BROADCAST $lhs, $dout), (BROADCAST $rhs, $dout)), $dout), $lhs),
        (REDUCE (MUL (NativeCodeCall<"dGreaterEqual($_builder, $0, $1)"> (BROADCAST $rhs, $dout), (BROADCAST $lhs, $dout)), $dout), $rhs),
    ]
>;

def MinimumToCore : Pattern<
    (Grad_MinimumOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (REDUCE (MUL (NativeCodeCall<"dGreaterEqual($_builder, $1, $0)"> (BROADCAST $lhs, $dout), (BROADCAST $rhs, $dout)), $dout), $lhs),
        (REDUCE (MUL (NativeCodeCall<"dGreaterEqual($_builder, $1, $0)"> (BROADCAST $rhs, $dout), (BROADCAST $lhs, $dout)), $dout), $rhs),
    ]
>;

def IntClampToCore : Pat<
    (Grad_ClampOp IT:$x, IT:$dout, I64Attr:$min_int, I64Attr:$max_int, F32Attr:$min_fp, F32Attr:$max_fp),
    (MUL (Tosa_SelectOp (NativeCodeCall<"intClampHelper($_builder, $0, $1, $2)"> $x, $min_int, $max_int), (ONESLIKE $x), (AD_ZeroslikeOp $x), (returnType $x)), $dout)
>;

def FloatClampToCore : Pat<
    (Grad_ClampOp FT:$x, FT:$dout, I64Attr:$min_int, I64Attr:$max_int, F32Attr:$min_fp, F32Attr:$max_fp),
    (MUL (Tosa_SelectOp (NativeCodeCall<"floatClampHelper($_builder, $0, $1, $2)"> $x, $min_fp, $max_fp), (ONESLIKE $x), (AD_ZeroslikeOp $x), (returnType $x)), $dout)
>;

def PowToCore : Pattern<
    (Grad_PowOp AT:$lhs, AT:$rhs, AT:$dout),
    [
        (REDUCE (MUL (MUL (Tosa_PowOp (BROADCAST:$x $lhs, $dout), (Tosa_SubOp (BROADCAST:$y $rhs, $dout), (ONESLIKE $y), (returnType $y)), (returnType $x)), $y), $dout), $lhs),
        (REDUCE (MUL (MUL (Tosa_PowOp $x, $y, (returnType $y)), (Tosa_LogOp $x, (returnType $x))), $dout), $rhs)
    ]
>;

def AvgPool2dToCore : Pat<
    (Grad_AvgPool2dOp:$dx $x, $dout, $kernel, $stride, $pad),
    (NativeCodeCall<"dAvgPool2d($_builder, $0)"> $dx)
>;

// def Conv2DToCore : Pattern<
//     (Grad_Conv2DOp:$dx $x, $dout, $weight, $bias, $dilation, $stride, $pad),
//     [
//         (NativeCodeCall<"dConv2DInput($_builder, $0)"> $dx__0),
//         (NativeCodeCall<"dConv2DBias($_builder, $0)"> $dx__1) // TODO: finish this
//     ]
// >;

def ReshapeToCore : Pat<
    (Grad_ReshapeOp:$dx $x, $dout, $new_shape),
    (NativeCodeCall<"dReshape($_builder, $0)"> $dx)
>;

def ReduceSumToCore : Pat<
    (Grad_ReduceSumOp $x, $dout),
    (BROADCAST $dout, $x)
>;

#endif // AD_CONVERSION_GRADTOCORE
