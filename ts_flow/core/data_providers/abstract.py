from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from ts_flow.core import Handler, T

__all__ = ["DataProvider", "T"]


class DataProvider(Handler[Any, T]):
    def __init__(self) -> None:
        super().__init__(source=None)

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass
