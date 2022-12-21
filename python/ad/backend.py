from typing import Any, List

import iree.compiler
import iree.runtime

__all__ = ["TosaExecutable"]


class Executable:
    def __init__(self, module: iree.runtime.VmModule):
        self.__module = module

    def exec(self, name: str, *args: Any) -> Any:
        return self.__module[name](*args)


class CompiledExecutable(Executable):
    def __init__(
        self, module: str, input_type: str, target_backends: List[str], config: str
    ):
        binary = iree.compiler.tools.compile_str(
            module, input_type=input_type, target_backends=target_backends
        )
        config = iree.runtime.system_api.Config(config)
        vm_instance = iree.runtime.VmInstance()
        vm_module = iree.runtime.VmModule.from_flatbuffer(vm_instance, binary)
        vm_object = iree.runtime.load_vm_module(vm_module, config)
        super().__init__(vm_object)


class TosaCpuLocalExecutable(CompiledExecutable):
    def __init__(self, module: str, name=None):
        self.__name = name
        super().__init__(module, "tosa", ["llvm-cpu"], "local-task")

    def exec(self, *args: Any) -> Any:
        if self.__name:
            return super().exec(self.__name, *args)
        return super().exec(args[0], *args[1:])


TosaExecutable = TosaCpuLocalExecutable
