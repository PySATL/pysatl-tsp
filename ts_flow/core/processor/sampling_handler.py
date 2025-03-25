from collections.abc import Iterator
from typing import Any, Callable, Optional

from ts_flow.core import Handler, T
from ts_flow.core.scrubber import OfflineSegmentationScrubber, OnlineSegmentationScrubber, ScrubberWindow

from .mapping_handler import MappingHandler


class OnlineSamplingHandler(Handler[T, T]):
    def __init__(self, sampling_rule: Callable[[ScrubberWindow[T]], bool], source: Optional[Handler[Any, T]] = None):
        super().__init__(source)
        self.sampling_rule = sampling_rule

    def __iter__(self) -> Iterator[T]:
        mapping_handler: MappingHandler[ScrubberWindow[T], T] = MappingHandler(map_func=lambda window: window[-1])
        pipeline = (
            OnlineSegmentationScrubber(segmentation_rule=self.sampling_rule, source=self.source) | mapping_handler
        )

        yield from pipeline


class OfflineSamplingHandler(Handler[T, T]):
    def __init__(
        self, sampling_rule: Callable[[ScrubberWindow[T]], list[int]], source: Optional[Handler[Any, T]] = None
    ):
        super().__init__(source)
        self.sampling_rule = sampling_rule

    def __iter__(self) -> Iterator[T]:
        mapping_handler: MappingHandler[ScrubberWindow[T], T] = MappingHandler(map_func=lambda window: window[-1])
        pipeline = (
            OfflineSegmentationScrubber(segmentation_rule=self.sampling_rule, source=self.source) | mapping_handler
        )

        yield from pipeline
