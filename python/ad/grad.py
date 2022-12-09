from ad.scalar import Scalar, gen_mlir
from mlir.ir import *
from typing import Union
import iree.compiler
import iree.runtime
import subprocess
import numpy as np
import os

# TODO: remove hardcoded part
ad = os.environ["AD_PATH"]
tmp_file = "tmp.mlir"


def grad_module(module: Module) -> str:
    with open(tmp_file, "w") as f:
        f.write(str(module))

    grad = (
        subprocess.Popen([ad, tmp_file, "--autodiff"], stdout=subprocess.PIPE)
        .stdout.read()
        .decode("utf-8")
    )

    subprocess.Popen(["rm", tmp_file])
    return grad


def run_module(module: str, func_name: str, *args: float) -> float:
    binary = iree.compiler.tools.compile_str(
        module, input_type="tosa", target_backends=["llvm-cpu"]
    )
    instance = iree.runtime.VmInstance()
    config = iree.runtime.system_api.Config("local-task")
    vm_module = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    vm_object = iree.runtime.load_vm_module(vm_module, config)
    args = map(lambda x: np.array([x], dtype=np.float32), args)
    res = vm_object[func_name](*args)

    if isinstance(res, tuple):
        return list(map(lambda x: x[0], res))

    return res[0]


__all__ = ["grad_module", "run_module"]
