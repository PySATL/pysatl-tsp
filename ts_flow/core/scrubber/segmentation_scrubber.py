from collections import deque
from collections.abc import Iterator
from typing import Any, Callable, Optional

from ts_flow.core import Handler, T

from .abstract import Scrubber, ScrubberWindow


class OfflineSegmentationScrubber(Scrubber[T]):
    def __init__(
        self, segmentation_rule: Callable[[ScrubberWindow[T]], list[int]], source: Optional[Handler[Any, T]] = None
    ):
        super().__init__(source)
        self.segmentation_rule = segmentation_rule

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        if self.source is None:
            raise ValueError("Source is not set")

        full_series_list = list(iter(self.source))
        full_series_deque = deque(full_series_list)
        series_window = ScrubberWindow(full_series_deque)
        change_points = self.segmentation_rule(series_window)
        segments = [0, *change_points, len(full_series_deque)]
        for start, end in zip(segments[:-1], segments[1:]):
            yield series_window[start:end]


class OnlineSegmentationScrubber(Scrubber[T]):
    def __init__(
        self,
        segmentation_rule: Callable[[ScrubberWindow[T]], bool],
        max_segment_size: int = 2**64,
        source: Optional[Handler[Any, T]] = None,
    ):
        super().__init__(source)
        self.segmentation_rule = segmentation_rule
        self.max_segment_size = max_segment_size

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        if self.source is None:
            raise ValueError("Source is not set")
        current_window: ScrubberWindow[T] = ScrubberWindow(deque())
        for index, item in enumerate(self.source):
            current_window.append(item, index)

            if self.segmentation_rule(current_window) or len(current_window) >= self.max_segment_size:
                yield current_window.copy()
                current_window.clear()

        if current_window:
            yield current_window.copy()
