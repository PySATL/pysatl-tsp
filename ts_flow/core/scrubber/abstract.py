from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Iterator, Any, overload, Union
from abc import abstractmethod

from ts_flow.core import T, Handler

__all__ = ["ScrubberWindow", "Scrubber", "T"]


class ScrubberWindow(Generic[T]):
    def __init__(self, values: list[T], indices: Optional[list[int]] = None) -> None:
        if indices is None:
            indices = list(range(len(values)))
        if len(values) != len(indices):
            raise ValueError("Values and indices of ScrubberWindow must be same length")
        self.values = values
        self.indices = indices

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> ScrubberWindow[T]: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, ScrubberWindow[T]]:
        match key:
            case int() as idx:
                return self.values[idx]

            case slice() as s:
                return ScrubberWindow(values=self.values[s], indices=self.indices[s])

            case _:
                raise TypeError(f"Unsupported key type: {type(key).__name__}")

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScrubberWindow):
            return NotImplemented
        return self.values == other.values and self.indices == other.indices


class Scrubber(Handler[T, ScrubberWindow[T]]):
    def __init__(self, source: Optional[Handler[Any, T]] = None) -> None:
        super().__init__(source)

    @abstractmethod
    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        pass
