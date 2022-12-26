import re

__all__ = ["convert_lines_to_iree", "convert_lines_to_ad"]

iree_statement = "linalg.init_tensor"
ad_statement = "bufferization.alloc_tensor()"


def convert_line_to_iree(line: str) -> str:
    tensor_shape = (
        line[line.find("<") :].replace("<", "").replace(">", "").split("x")[:-1]
    )
    array_attr = f"[{', '.join(tensor_shape)}]"
    return line.replace(ad_statement, f"{iree_statement} {array_attr}")


def convert_lines_to_iree(lines: str) -> str:
    if not ad_statement in lines:
        return lines
    lines = lines.split("\n")
    for i in range(len(lines)):
        lines[i] = convert_line_to_iree(lines[i])
    return "\n".join(lines)


def convert_lines_to_ad(lines: str) -> str:
    if not iree_statement in lines:
        return lines
    return re.sub(f"{iree_statement} \[.*\]", ad_statement, lines)
