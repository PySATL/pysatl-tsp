from collections.abc import Iterator
from typing import Any, Optional

from ts_flow.core import Handler

from .abstract import Scrubber, ScrubberWindow, T


class LinearScrubber(Scrubber[T]):
    def __init__(
        self, window_length: int = 100, shift_factor: float = 1.0 / 3.0, source: Optional[Handler[Any, T]] = None
    ) -> None:
        super().__init__(source)
        self._window_length = window_length
        self._shift: int = max(1, int(shift_factor * window_length))
        self._buffer: ScrubberWindow[T] = ScrubberWindow()

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        if self.source is None:
            raise ValueError("Source is not set")

        for i, item in enumerate(self.source):
            self._buffer.append(item, i)
            while len(self._buffer) >= self._window_length:
                yield self._buffer[: self._window_length]

                for _ in range(self._shift):
                    if self._buffer:
                        self._buffer.popleft()
