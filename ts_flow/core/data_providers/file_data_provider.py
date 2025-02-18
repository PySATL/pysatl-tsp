from .abstract import DataProvider
from typing import Iterator, Callable, TypeVar

X = TypeVar("X")


class FileDataProvider(DataProvider[X]):
    def __init__(self, filename: str, handler: Callable[[str], X]) -> None:
        super().__init__()
        self.filename = filename
        self.handler = handler

    def __iter__(self) -> Iterator[X]:
        with open(self.filename, "r") as f:
            for line in f:
                yield self.handler(line)
