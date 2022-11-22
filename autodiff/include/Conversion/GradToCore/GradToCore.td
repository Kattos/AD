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

class ValidTypes : AnyTypeOf<[AnyTensor]>;

// alias to simplify coding
def VT : ValidTypes;

def AddToCore : Pattern<
    (Grad_AddOp VT:$lhs, VT:$rhs, VT:$dout),
    [
        (AD_ReduceOp (MUL (AD_OneslikeOp (AD_BroadcastOp $lhs, $dout)), $dout), $lhs),
        (AD_ReduceOp (MUL (AD_OneslikeOp (AD_BroadcastOp $rhs, $dout)), $dout), $rhs),
    ]
>;

def SubToCore : Pattern<
    (Grad_SubOp VT:$lhs, VT:$rhs, VT:$dout),
    [
        (AD_ReduceOp (MUL (AD_OneslikeOp (AD_BroadcastOp $lhs, $dout)), $dout), $lhs),
        (AD_ReduceOp (MUL (NEGATE (AD_OneslikeOp (AD_BroadcastOp $rhs, $dout))), $dout), $rhs)
    ]
>;

def MulToCore : Pattern<
    (Grad_MulOp VT:$lhs, VT:$rhs, VT:$dout),
    [
        (AD_ReduceOp (MUL (AD_BroadcastOp $rhs, $dout), $dout), $lhs),
        (AD_ReduceOp (MUL (AD_BroadcastOp $lhs, $dout), $dout), $rhs)
    ]
>;

def ExpToCore : Pat<
    (Grad_ExpOp VT:$x, VT:$dout),
    (MUL (EXP $x), $dout)
>;

def LogToCore : Pat<
    (Grad_LogOp VT:$x, VT:$dout),
    (MUL (RECIPROCAL $x), $dout)
>;

def RsqrtToCore : Pat<
    (Grad_RsqrtOp VT:$x, VT:$dout),
    (MUL (NativeCodeCall<"drsqrt($_builder, $0)"> $x), $dout)
>;

// \derivative{tanh(x)}{x} = \frac{4}{(e^x + e^{-x})^2}
def TanhToCore : Pat<
    (Grad_TanhOp VT:$x, VT:$dout),
    (MUL (ADD (ADD:$half (RECIPROCAL:$quarter (MUL:$dominant (ADD:$sqrt (EXP $x), (EXP (NEGATE $x))), $sqrt)), $quarter), $half), $dout)
>;

def NegateToCore : Pat<
    (Grad_NegateOp VT:$x, VT:$dout),
    (MUL (NEGATE (AD_OneslikeOp $x)), $dout)
>;

def ReciprocalToCore : Pat<
    (Grad_ReciprocalOp VT:$x, VT:$dout),
    (MUL (NEGATE (RECIPROCAL (MUL $x, $x))), $dout)
>;

// \derivative{sigmoid(x)}{x} = \frac{e^{-x}}{(1 + e^{-x})^2}
def SigmoidToCore : Pat<
    (Grad_SigmoidOp VT: $x, VT:$dout),
    (MUL (MUL (RECIPROCAL (MUL (ADD:$sqrt (EXP:$ex (NEGATE $x)), (AD_OneslikeOp $ex)), $sqrt)), $ex), $dout)
>;

#endif // AD_CONVERSION_GRADTOCORE
