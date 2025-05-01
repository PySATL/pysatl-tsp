from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Iterator
from itertools import islice
from typing import Any, Generic, overload

from pysatl_tsp.core import Handler, T

__all__ = ["Scrubber", "ScrubberWindow", "T"]


class ScrubberWindow(Generic[T]):
    def __init__(self, values: deque[T] | None = None, indices: deque[int] | None = None) -> None:
        if values is None:
            values = deque()
        if indices is None:
            indices = deque(range(len(values)))
        if len(values) != len(indices):
            raise ValueError("Values and indices of ScrubberWindow must be same length")
        self.values = values
        self.indices = indices

    def append(self, value: T, index: int | None = None) -> None:
        self.values.append(value)
        if index is None:
            index = len(self)
        self.indices.append(index)

    def popleft(self) -> None:
        self.values.popleft()
        self.indices.popleft()

    def clear(self) -> None:
        self.values.clear()
        self.indices.clear()

    def copy(self) -> ScrubberWindow[T]:
        return ScrubberWindow(self.values.copy(), self.indices.copy())

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> ScrubberWindow[T]: ...

    def __getitem__(self, key: int | slice) -> T | ScrubberWindow[T]:
        match key:
            case int() as idx:
                return self.values[idx]

            case slice() as s:
                return ScrubberWindow(
                    values=deque(islice(self.values, s.start, s.stop)),
                    indices=deque(islice(self.indices, s.start, s.stop)),
                )

            case _:
                raise TypeError(f"Unsupported key type: {type(key).__name__}")

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScrubberWindow):
            return NotImplemented
        return self.values == other.values and self.indices == other.indices

    def __repr__(self) -> str:
        return f"ScrubberWindow(values: {self.values}, indices: {self.indices})"

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)


class Scrubber(Handler[T, ScrubberWindow[T]]):
    def __init__(self, source: Handler[Any, T] | None = None) -> None:
        super().__init__(source)

    @abstractmethod
    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        pass
