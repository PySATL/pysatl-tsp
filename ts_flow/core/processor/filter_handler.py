from collections import deque
from collections.abc import Iterator
from typing import Any, Callable, Optional

from ts_flow.core import Handler, T, U
from ts_flow.core.scrubber import ScrubberWindow


class OnlineFilterHandler(Handler[T, U]):
    def __init__(
        self,
        filter_func: Callable[[ScrubberWindow[T], Any], U],
        filter_config: Any = None,
        source: Optional[Handler[Any, T]] = None,
    ):
        super().__init__(source)
        self.filter_func = filter_func
        self.filter_config = filter_config

    def __iter__(self) -> Iterator[U]:
        if self.source is None:
            raise ValueError("Source is not set")

        self._history: ScrubberWindow[T] = ScrubberWindow()

        for item in self.source:
            self._history.append(item)
            yield self.filter_func(self._history, self.filter_config)


class OfflineFilterHandler(Handler[T, U]):
    def __init__(
        self,
        filter_func: Callable[[ScrubberWindow[T], Any], list[U]],
        filter_config: Any = None,
        source: Optional[Handler[Any, T]] = None,
    ):
        super().__init__(source)
        self.filter_func = filter_func
        self.filter_config = filter_config

    def __iter__(self) -> Iterator[U]:
        if self.source is None:
            raise ValueError("Source is not set")

        full_series = ScrubberWindow(deque(self.source))
        filtered_series = self.filter_func(full_series, self.filter_config)

        yield from filtered_series
