from __future__ import annotations
from enum import Enum
from typing import List, Union
import os
import subprocess
from ad.compat import convert_lines_to_iree

from mlir.ir import Module

from ad.backend import Executable, TosaExecutable

__all__ = ["autodiff", "TosaIR"]

assert "AD_PATH" in os.environ, "AD_PATH is not initialized"


class ADPass(Enum):
    AUTODIFF = "--autodiff"
    UNREGISTERED = "--allow-unregistered-dialect"


def optimize(primal: str, params: Union[ADPass, List[ADPass]]) -> str:
    # TODO: do not I/O here

    opt = os.environ["AD_PATH"]
    buffer = "tmp.mlir"

    with open(buffer, "w") as f:
        f.write(primal)

    params = (
        params.value
        if not isinstance(params, list)
        else list(map(lambda x: x.value, params))
    )

    compiled = (
        subprocess.Popen([opt, buffer, *params], stdout=subprocess.PIPE)
        .stdout.read()
        .decode("utf-8")
    )

    subprocess.Popen(["rm", buffer])
    return compiled


def autodiff(primal: str) -> str:
    return optimize(primal, [ADPass.AUTODIFF, ADPass.UNREGISTERED])


class Representation:
    def __init__(self, module: Union[str, Module]):
        # TODO: validate module
        self._module = str(module)

    def __str__(self):
        return self._module

    def to_executable(self) -> Executable:
        return None


IR = Representation


class TosaIR(IR):
    def __init__(self, module: Union[str, Module]):
        super().__init__(module)
        self._executable: TosaExecutable = None

    def __str__(self):
        return super().__str__()

    def __eq__(self, other):
        if not isinstance(other, TosaIR):
            return False
        return self._module == other._module

    def dump(self):
        print(self._module)

    def to_executable(self, name=None) -> TosaExecutable:
        return None


class IreeIR(TosaIR):
    def __init___(self, module: Union[str, Module, TosaIR]):
        super().__init__(module)

    def to_executable(self, name=None) -> TosaExecutable:
        if self._executable:
            return self._executable

        if name:
            return TosaExecutable(self._module, name)
        # parse func name
        trim = self._module[self._module.find("func.func @") + len("func.func @") :]
        fn = trim[: trim.find("(")]
        self._executable = TosaExecutable(self._module, fn)
        return self._executable


class AdIR(TosaIR):
    def __init__(self, module: Union[str, Module, TosaIR]):
        super().__init__(module)

    def grad(self) -> AdIR:
        return AdIR(autodiff(self._module))

    def to_executable(self, name=None) -> TosaExecutable:
        if self._executable:
            return self._executable
        iree = IreeIR(convert_lines_to_iree(self._module))
        self._executable = iree.to_executable(name)
        return self._executable
