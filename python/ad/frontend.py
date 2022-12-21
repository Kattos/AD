from __future__ import annotations
from typing import Union
from ad.compatible import convert_lines_to_ad, convert_lines_to_iree

from ad.intermediate import IR, AdIR, IreeIR, TosaIR


__all__ = ["FileParser", "IreeParser", "AdParser"]


class Parser:
    def to_ir(self) -> IR:
        return None


class TosaParser(Parser):
    def __init__(self, content: str):
        self._content = content

    def __str__(self):
        return self._content

    def __eq__(self, other):
        if not isinstance(other, TosaParser):
            return False
        return self._content == other._content

    def to_ir(self) -> TosaIR:
        return TosaIR(self._content)


class FileParser(TosaParser):
    def __init__(self, filepath: str):
        with open(filepath) as f:
            super().__init__(f.read())


class IreeParser(TosaParser):
    def __init__(self, content: Union[str, TosaParser]):
        super().__init__(str(content))

    def cast_to_ad(self) -> AdParser:
        return AdParser(self._content)

    def cast(self) -> AdParser:
        return self.cast_to_ad()

    def to_iree_ir(self) -> IreeIR:
        return IreeIR(convert_lines_to_iree(self._content))

    def to_ir(self) -> IreeIR:
        return self.to_iree_ir()


class AdParser(TosaParser):
    def __init__(self, content: Union[str, TosaParser]):
        super().__init__(str(content))

    def cast_to_iree(self) -> IreeParser:
        return IreeParser(self._content)

    def cast(self) -> AdParser:
        return self.cast_to_iree()

    def to_ad_ir(self) -> AdIR:
        return AdIR(convert_lines_to_ad(self._content))

    def to_ir(self) -> AdIR:
        return self.to_ad_ir()
