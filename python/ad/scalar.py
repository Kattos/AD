from __future__ import annotations
from mlir.ir import *
from mlir.dialects import tosa, arith, tensor, func
from typing import Callable, List, Optional, Union


class Scalar:
    def __init__(self, value: Value):
        self.value = value

    @classmethod
    def init(cls, context: Context, location: Location):
        cls.__ctx = context
        cls.__loc = location
        cls.__f32_type = F32Type.get(context)
        cls.__i32_type = IntegerType.get_signless(32)
        cls.__f32_scalar_type = RankedTensorType.get(
            shape=[1], element_type=cls.__f32_type, loc=location
        )

    @classmethod
    def get(cls, value: Union[float, Value]) -> Scalar:
        if isinstance(value, Value):
            return Scalar(value)
        cst = arith.ConstantOp(value=value, result=cls.__f32_type).result
        scalar = tensor.FromElementsOp.build_generic(
            operands=[cst], results=[cls.__f32_scalar_type]
        ).result
        return Scalar(scalar)

    @classmethod
    def __do_tosa(cls, op: Operation, *operand: Scalar) -> Scalar:
        return Scalar.__do_tosa_with_attrs(op, None, *operand)

    @classmethod
    def __do_tosa_with_attrs(
        cls, op: Operation, attributes: Optional[dict] = None, *operand: Scalar
    ) -> Scalar:
        with cls.__ctx, cls.__loc:
            result = op.build_generic(
                operands=list(map(lambda x: x.value, operand)),
                results=[cls.__f32_scalar_type],
                attributes=attributes,
            ).result
        return Scalar(result)

    @classmethod
    def get_ir_type(cls, value: Union[float, int, List, Value]) -> Type:
        if isinstance(value, float):
            return cls.__f32_type
        elif isinstance(value, int):
            return cls.__i32_type
        elif isinstance(value, list):
            return cls.__f32_scalar_type
        else:
            return value.type

    def __str__(self) -> str:
        return self.value.__str__()

    def __add__(self, another: Scalar) -> Scalar:
        return Scalar.__do_tosa(tosa.AddOp, self, another)

    def __sub__(self, another: Scalar) -> Scalar:
        return Scalar.__do_tosa(tosa.SubOp, self, another)

    def __pow__(self, another: Scalar) -> Scalar:
        return Scalar.__do_tosa(tosa.PowOp, self, another)

    def __mul__(self, another: Scalar) -> Scalar:
        shift = IntegerAttr.get(self.__i32_type, 0)
        return Scalar.__do_tosa_with_attrs(tosa.MulOp, {"shift": shift}, self, another)

    def __abs__(self) -> Scalar:
        return Scalar.__do_tosa(tosa.AbsOp, self)

    def exp(self) -> Scalar:
        return Scalar.__do_tosa(tosa.ExpOp, self)

    def log(self) -> Scalar:
        return Scalar.__do_tosa(tosa.LogOp, self)

    def rsqrt(self) -> Scalar:
        return Scalar.__do_tosa(tosa.RsqrtOp, self)

    def tanh(self) -> Scalar:
        return Scalar.__do_tosa(tosa.TanhOp, self)

    def negate(self) -> Scalar:
        return Scalar.__do_tosa(tosa.NegateOp, self)

    def reciprocal(self) -> Scalar:
        return Scalar.__do_tosa(tosa.ReciprocalOp, self)

    def sigmoid(self) -> Scalar:
        return Scalar.__do_tosa(tosa.SigmoidOp, self)


def pyfunc_to_mlir(
    context: Context,
    location: Location,
    inputs: int,
    outputs: int,
    fn_name: str,
    fn: Callable,
) -> Module:
    with context as ctx, location as loc:
        Scalar.init(ctx, loc)
        module = Module.create()
        scalar_type = Scalar.get_ir_type([])
        with InsertionPoint(module.body):
            mlir_func = func.FuncOp(
                fn_name, ([scalar_type] * inputs, [scalar_type] * outputs)
            )
            block = mlir_func.add_entry_block()
            with InsertionPoint(block):
                args = map(lambda x: Scalar(x), mlir_func.arguments)
                result_scalars = fn(*args) if outputs > 1 else [fn(*args)]
                result = list(map(lambda x: x.value, result_scalars))
                func.ReturnOp.build_generic(operands=result, results=[])
        return module


__all__ = ["Scalar", "pyfunc_to_mlir"]
