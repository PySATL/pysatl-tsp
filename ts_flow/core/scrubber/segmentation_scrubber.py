from abc import ABC, abstractmethod
from .abstract import Scrubber, ScrubberWindow
from typing import Callable, List, Optional, Iterator, Any, Generic
from ts_flow.core import T, Handler


class SegmentationStrategy(ABC, Generic[T]):
    @abstractmethod
    def generate_segments(self, source: Iterator[T]) -> Iterator[ScrubberWindow[T]]:
        pass


class OfflineStrategy(SegmentationStrategy[T]):
    def __init__(self, segmentation_rule: Callable[[List[T]], List[int]]):
        self.segmentation_rule = segmentation_rule

    def generate_segments(self, source: Iterator[T]) -> Iterator[ScrubberWindow[T]]:
        full_series = list(source)
        change_points = self.segmentation_rule(full_series)
        segments = [0] + change_points + [len(full_series)]
        for start, end in zip(segments[:-1], segments[1:]):
            yield ScrubberWindow(values=full_series[start:end], indices=list(range(start, end)))


class OnlineStrategy(SegmentationStrategy[T]):
    def __init__(self, segmentation_rule: Callable[[List[T]], bool], max_segment_size: int = 100):
        self.segmentation_rule = segmentation_rule
        self.max_segment_size = max_segment_size

    def generate_segments(self, source: Iterator[T]) -> Iterator[ScrubberWindow[T]]:
        buffer = []
        indices = []
        current_idx = 0
        for item in source:
            buffer.append(item)
            indices.append(current_idx)
            current_idx += 1

            if self.segmentation_rule(buffer) or len(buffer) >= self.max_segment_size:
                yield ScrubberWindow(values=buffer.copy(), indices=indices.copy())
                buffer.clear()
                indices.clear()

        if buffer:
            yield ScrubberWindow(values=buffer.copy(), indices=indices.copy())


class SegmentationScrubber(Scrubber[T]):
    def __init__(self, strategy: SegmentationStrategy[T], source: Optional[Handler[Any, T]] = None):
        super().__init__(source)
        self.strategy = strategy

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        if self.source is None:
            raise ValueError("Source is not set")
        return self.strategy.generate_segments(iter(self.source))
