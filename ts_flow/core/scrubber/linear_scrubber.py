from ts_flow.core import Handler
from typing import Optional, Any, Iterator
from itertools import islice

from collections import deque
from .abstract import Scrubber, ScrubberWindow, T


class LinearScrubber(Scrubber[T]):
    def __init__(
        self, window_length: int = 100, shift_factor: float = 1.0 / 3.0, source: Optional[Handler[Any, T]] = None
    ) -> None:
        super().__init__(source)
        self._window_length = window_length
        self._shift: int = max(1, int(shift_factor * window_length))
        self._values_buffer: deque[T] = deque(maxlen=window_length + self._shift)
        self._indices_buffer: deque[int] = deque(maxlen=window_length + self._shift)
        self._current_index = 0

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        if self.source is None:
            raise ValueError("Source is not set")

        for item in self.source:
            self._values_buffer.append(item)
            self._indices_buffer.append(self._current_index)
            while len(self._values_buffer) >= self._window_length:
                values = list(islice(self._values_buffer, 0, self._window_length))
                indices = list(islice(self._indices_buffer, 0, self._window_length))
                yield ScrubberWindow(values, indices)

                for _ in range(self._shift):
                    if self._values_buffer:
                        self._values_buffer.popleft()
                    if self._indices_buffer:
                        self._indices_buffer.popleft()
            self._current_index += 1
