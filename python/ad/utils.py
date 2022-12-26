from functools import wraps
from time import perf_counter, process_time, process_time_ns
from typing import Callable
import re

__all__ = ["timer_s", "timer_us", "timer_ns"]


def timer_general(func: Callable, counter: Callable, unit: str) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = counter()
        res = func(*args, **kwargs)
        time_end = counter()
        time_cost = time_end - time_start
        print(f"{func.__name__} : {time_cost} {unit}")
        return res, time_cost

    return wrapper


def timer_s(func: Callable) -> Callable:
    return timer_general(func, process_time, "s")


def timer_us(func: Callable) -> Callable:
    return timer_general(func, perf_counter, "us")


def timer_ns(func: Callable) -> Callable:
    return timer_general(func, process_time_ns, "ns")


alloc_statement = "bufferization.alloc_tensor()"
init_statement = "linalg.init_tensor"

# transform `bufferizaion.alloc_tensor` to `linalg.init` for compatibility
def alloc_to_init(line: str) -> str:
    if not has_alloc_statement(line):
        return line
    tensor_shape = (
        line[line.find("<") :].replace("<", "").replace(">", "").split("x")[:-1]
    )
    array_attr = f"[{', '.join(tensor_shape)}]"
    return line.replace(alloc_statement, f"{init_statement} {array_attr}")


def init_to_alloc(line: str) -> str:
    if not has_init_statement(line):
        return line
    return re.sub(f"{init_statement} \[.*\]", alloc_statement, line)


def all_alloc_to_init(lines: str) -> str:
    lines = lines.split("\n")
    for i in range(len(lines)):
        lines[i] = alloc_to_init(lines[i])
    return "\n".join(lines)


def all_init_to_alloc(lines: str) -> str:
    return init_to_alloc(lines)


def has_init_statement(lines: str) -> bool:
    return init_statement in lines


def has_alloc_statement(lines: str) -> bool:
    return alloc_statement in lines
