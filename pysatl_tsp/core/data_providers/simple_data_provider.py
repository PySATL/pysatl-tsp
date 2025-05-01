from collections.abc import Iterable, Iterator

from .abstract import DataProvider, T


class SimpleDataProvider(DataProvider[T]):
    def __init__(self, data: Iterable[T]) -> None:
        super().__init__()
        self.data = data

    def __iter__(self) -> Iterator[T]:
        yield from self.data
