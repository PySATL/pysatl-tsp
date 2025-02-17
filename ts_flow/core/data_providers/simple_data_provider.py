from .abstract import DataProvider, T
from typing import Iterable, Iterator


class SimpleDataProvider(DataProvider[T]):
    def __init__(self, data: Iterable[T]) -> None:
        super().__init__()
        self.data = data

    def __iter__(self) -> Iterator[T]:
        yield from self.data
