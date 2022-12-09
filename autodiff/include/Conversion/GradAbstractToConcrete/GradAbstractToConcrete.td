#ifndef AD_CONVERSION_GRADABSTRACTTOCONCRETE
#define AD_CONVERSION_GRADABSTRACTTOCONCRETE

include "mlir/IR/PatternBase.td"
include "Dialect/Grad/IR/Grad.td"

class AbstractUnaryPat<dag result, dag pred> : Pat<
    (Grad_AbstractUnaryOp:$unary $x, $dout, $op),
    result,
    [pred]
>;

def UnaryToAbs : AbstractUnaryPat<
    (Grad_AbsOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::AbsOp::getOperationName()">> $op)
>;

def UnaryToExp : AbstractUnaryPat<
    (Grad_ExpOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::ExpOp::getOperationName()">> $op)
>;

def UnaryToLog : AbstractUnaryPat<
    (Grad_LogOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::LogOp::getOperationName()">> $op)
>;

def UnaryToRsqrt : AbstractUnaryPat<
    (Grad_RsqrtOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::RsqrtOp::getOperationName()">> $op)
>;

def UnaryToTanh : AbstractUnaryPat<
    (Grad_TanhOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::TanhOp::getOperationName()">> $op)
>;

def UnaryToNegate : AbstractUnaryPat<
    (Grad_NegateOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::NegateOp::getOperationName()">> $op)
>;

def UnaryToReciprocal : AbstractUnaryPat<
    (Grad_ReciprocalOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::ReciprocalOp::getOperationName()">> $op)
>;

def UnaryToSigmoid : AbstractUnaryPat<
    (Grad_SigmoidOp $x, $dout),
    (Constraint<CPred<"$0.str() == tosa::SigmoidOp::getOperationName()">> $op)
>;

def UnaryToClamp : AbstractUnaryPat<
    (NativeCodeCall<"toConcreteWithAttrs<grad::ClampOp>($_builder, $0)"> $unary),
    (Constraint<CPred<"$0.str() == tosa::ClampOp::getOperationName()">> $op)
>;

def UnaryToAvgPool2d : AbstractUnaryPat<
    (NativeCodeCall<"toConcreteWithAttrs<grad::AvgPool2dOp>($_builder, $0)"> $unary),
    (Constraint<CPred<"$0.str() == tosa::AvgPool2dOp::getOperationName()">> $op)
>;

class AbstractBinaryPat<dag result, dag pred> : Pat<
    (Grad_AbstractBinaryOp:$binary $lhs, $rhs, $dout, $op),
    result,
    [pred]
>;

def BinaryToAdd : AbstractBinaryPat<
    (Grad_AddOp $lhs, $rhs, $dout),
    (Constraint<CPred<"$0.str() == tosa::AddOp::getOperationName()">> $op)
>;

def BinaryToSub : AbstractBinaryPat<
    (Grad_SubOp $lhs, $rhs, $dout),
    (Constraint<CPred<"$0.str() == tosa::SubOp::getOperationName()">> $op)
>;

def BinaryToMul : AbstractBinaryPat<
    (Grad_MulOp $lhs, $rhs, $dout),
    (Constraint<CPred<"$0.str() == tosa::MulOp::getOperationName()">> $op)
>;

def BinaryToPow : AbstractBinaryPat<
    (Grad_PowOp $lhs, $rhs, $dout),
    (Constraint<CPred<"$0.str() == tosa::PowOp::getOperationName()">> $op)
>;


#endif // AD_CONVERSION_GRADABSTRACTTOCONCRETE
