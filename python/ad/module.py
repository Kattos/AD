from __future__ import annotations
from typing import Any, Optional, Tuple, Union
from mlir.ir import Module as MLIRModule

import iree.compiler
import iree.runtime
import numpy as np
import os
import subprocess


class Module:
    def __init__(self, module: Union[str, MLIRModule], func_name: str):
        if isinstance(module, MLIRModule):
            module = str(module)
        binary = iree.compiler.tools.compile_str(
            module, input_type="tosa", target_backends=["llvm-cpu"]
        )
        vm_instance = iree.runtime.VmInstance()
        config = iree.runtime.system_api.Config("local-task")
        vm_module = iree.runtime.VmModule.from_flatbuffer(vm_instance, binary)
        vm_object = iree.runtime.load_vm_module(vm_module, config)
        self._module = module
        self._func_name = func_name
        self.__vm_object = vm_object

    def __str__(self):
        return self._module

    def run(self, *args: Any) -> Optional[Union[Tuple[Any], Any]]:
        return self.__vm_object[self._func_name](*args)


class ScalarModule(Module):
    def __init__(self, module: Union[str, MLIRModule], func_name: str):
        super().__init__(module, func_name)

    def run(self, *args: float) -> Optional[Union[Tuple[float], float]]:
        for i, arg in enumerate(args):
            assert isinstance(arg, float), "Argument must be a float"

        res = super().run(*map(lambda x: np.array([x], dtype=np.float32), args))

        if isinstance(res, tuple):
            return list(map(lambda x: x[0], res))

        return res[0]

    def grad(self) -> ScalarModule:
        assert os.environ["AD_PATH"], "AD_PATH is not initialized"

        ad = os.environ["AD_PATH"]
        tmp = "tmp.mlir"
        params = "--autodiff"

        with open(tmp, "w") as f:
            f.write(self._module)

        grad = (
            subprocess.Popen([ad, tmp, params], stdout=subprocess.PIPE)
            .stdout.read()
            .decode("utf-8")
        )

        subprocess.Popen(["rm", tmp])
        return ScalarModule(grad, f"diff_{self._func_name}")

__all__ = ["Module", "ScalarModule"]
            