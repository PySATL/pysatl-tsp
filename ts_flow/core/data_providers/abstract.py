from ts_flow.core import Handler, T
from abc import abstractmethod
from typing import Iterator, Any

__all__ = ["DataProvider", "T"]


class DataProvider(Handler[Any, T]):
    def __init__(self) -> None:
        super().__init__(source=None)

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass
