from collections import deque
from collections.abc import Iterator
from typing import Any, Callable

from pysatl_tsp.core import Handler, T

from .abstract import Scrubber, ScrubberWindow


class OfflineSegmentationScrubber(Scrubber[T]):
    """A scrubber that segments time series data based on changepoints in batch mode.

    This scrubber processes the entire input data in a batch (offline) mode and
    segments it according to a provided segmentation rule. The rule identifies
    changepoints in the data, which are then used to create non-overlapping segments.

    This approach is suitable for scenarios where the entire dataset is available upfront
    and the segmentation logic requires global context or multiple passes over the data.

    :param segmentation_rule: Function that analyzes the complete series and returns a list of changepoint indices
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source with synthetic pattern
        data = [1, 1, 2, 2, 5, 5, 5, 1, 1, 1, 6, 6, 6, 6]
        data_source = SimpleDataProvider(data)


        # Define a simple variance-based segmentation rule
        def find_changepoints(window: ScrubberWindow[int]) -> list[int]:
            changepoints = []
            # Simple detection of value changes
            for i in range(1, len(window)):
                if abs(window[i] - window[i - 1]) > 2:  # Threshold for change
                    changepoints.append(i)
            return changepoints


        # Create the segmentation scrubber
        segmenter = OfflineSegmentationScrubber(segmentation_rule=find_changepoints, source=data_source)

        # Process the segments
        for segment in segmenter:
            print(f"Segment values: {list(segment.values)}")

        # Output:
        # Segment values: [1, 1, 2, 2]
        # Segment values: [5, 5, 5]
        # Segment values: [1, 1, 1]
        # Segment values: [6, 6, 6, 6]
        ```
    """

    def __init__(
        self, segmentation_rule: Callable[[ScrubberWindow[T]], list[int]], source: Handler[Any, T] | None = None
    ):
        """Initialize an offline segmentation scrubber.

        :param segmentation_rule: Function that analyzes the complete series and returns a list of changepoint indices
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.segmentation_rule = segmentation_rule

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        """Create an iterator that yields segments based on detected changepoints.

        This method collects all data from the source, applies the segmentation rule
        to identify changepoints, and then yields segments between the detected changepoints.

        :return: Iterator yielding ScrubberWindow instances for each segment
        :raises ValueError: If no source has been set
        """
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
    """A scrubber that segments time series data in real-time based on a condition.

    This scrubber processes data points sequentially (online mode) and segments
    the time series whenever a specified condition is met or a maximum segment size
    is reached. It's designed for streaming data where segments need to be identified
    in real-time without waiting for the complete dataset.

    :param segmentation_rule: Function that evaluates the current window and returns True when a segment should end
    :param max_segment_size: Maximum number of points in a segment before forcing a split, defaults to 2^64
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source with streaming values
        data = [1, 1, 2, 3, 8, 9, 8, 2, 2, 3, 10, 10, 9, 9]
        data_source = SimpleDataProvider(data)


        # Define a threshold-based segmentation rule
        def detect_jump(window: ScrubberWindow[int]) -> bool:
            if len(window) < 2:
                return False

            # Detect a large jump in values
            last_value = window[-1]
            prev_value = window[-2]
            return abs(last_value - prev_value) > 3


        # Create the online segmentation scrubber
        segmenter = OnlineSegmentationScrubber(
            segmentation_rule=detect_jump,
            max_segment_size=5,  # Force segmentation after 5 points if no jump detected
            source=data_source,
        )

        # Process the segments as they're detected
        for segment in segmenter:
            print(f"Segment values: {list(segment.values)}")

        # Output:
        # Segment values: [1, 1, 2, 3, 8]  # Split due to jump from 3 to 8 and max size
        # Segment values: [9, 8, 2]        # Split due to jump from 8 to 2
        # Segment values: [2, 3, 10]       # Split due to jump from 3 to 10
        # Segment values: [10, 9, 9]       # Remaining data
        ```
    """

    def __init__(
        self,
        segmentation_rule: Callable[[ScrubberWindow[T]], bool],
        max_segment_size: int = 2**64,
        source: Handler[Any, T] | None = None,
    ):
        """Initialize an online segmentation scrubber.

        :param segmentation_rule: Function that evaluates the current window and returns True when a segment should end
        :param max_segment_size: Maximum number of points in a segment before forcing a split, defaults to 2^64
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.segmentation_rule = segmentation_rule
        self.max_segment_size = max_segment_size

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        """Create an iterator that yields segments as they're detected in real-time.

        This method processes data points one by one, accumulating them in a buffer
        and checking after each addition whether the segmentation condition is met
        or the maximum segment size is reached.

        :return: Iterator yielding ScrubberWindow instances for each detected segment
        :raises ValueError: If no source has been set
        """
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
