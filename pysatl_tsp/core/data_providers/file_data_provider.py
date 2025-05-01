from collections.abc import Iterator
from typing import Callable, TypeVar

from .abstract import DataProvider

X = TypeVar("X")


class FileDataProvider(DataProvider[X]):
    def __init__(self, filename: str, handler: Callable[[str], X]) -> None:
        super().__init__()
        self.filename = filename
        self.handler = handler

    def __iter__(self) -> Iterator[X]:
        with open(self.filename) as f:
            for line in f:
                yield self.handler(line)
