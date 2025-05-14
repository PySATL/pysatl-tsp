from collections.abc import Iterator
from typing import Any, Callable

from pysatl_tsp.core import Handler

from .abstract import Scrubber, ScrubberWindow, T


class SlidingScrubber(Scrubber[T]):
    """A flexible scrubber that creates sliding windows of time series data based on custom conditions.

    This scrubber allows defining custom conditions for when to emit a window and how far to
    slide the window after each emission. It accumulates data points in a buffer and yields
    the current window whenever the take condition evaluates to True.

    :param take_condition: Function that determines when to emit the current window
    :param shift: Number of points to shift the window after each emission
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source
        data_source = SimpleDataProvider([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Emit windows when they contain exactly 3 elements, shift by 2
        condition = lambda window: len(window) == 3
        scrubber = SlidingScrubber(take_condition=condition, shift=2, source=data_source)

        # Process the windows
        for window in scrubber:
            print(f"Window values: {list(window.values)}")

        # Output:
        # Window values: [1, 2, 3]
        # Window values: [3, 4, 5]
        # Window values: [5, 6, 7]
        # Window values: [7, 8, 9]
        # Window values: [9, 10]

        # Create a scrubber that emits windows based on their sum
        sum_condition = lambda window: sum(window.values) >= 10
        sum_scrubber = SlidingScrubber(take_condition=sum_condition, shift=1, source=data_source)

        for window in sum_scrubber:
            print(f"Window with sum >= 10: {list(window.values)}, sum: {sum(window.values)}")
        ```
    """

    def __init__(
        self, take_condition: Callable[[ScrubberWindow[T]], bool], shift: int, source: Handler[Any, T] | None = None
    ) -> None:
        """Initialize a sliding scrubber with custom condition and shift.

        :param take_condition: Function that determines when to emit the current window
        :param shift: Number of points to shift the window after each emission
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self._shift = shift
        self._buffer: ScrubberWindow[T] = ScrubberWindow()
        self._take_condition = take_condition

    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        """Create an iterator that yields windows based on the take condition.

        This method accumulates data points in a buffer and yields the current window
        whenever the take condition evaluates to True. After yielding, it shifts
        the window by the specified number of points.

        :return: Iterator yielding ScrubberWindow instances
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        for i, item in enumerate(self.source):
            self._buffer.append(item, i)
            if self._take_condition(self._buffer):
                yield self._buffer[:]

                for _ in range(self._shift):
                    if self._buffer:
                        self._buffer.popleft()


class LinearScrubber(SlidingScrubber[T]):
    """A scrubber that creates fixed-size sliding windows with configurable overlap.

    This is a specialized sliding scrubber that emits windows of a fixed size and
    allows controlling the overlap between consecutive windows through a shift factor.

    :param window_length: Number of points in each window, defaults to 100
    :param shift_factor: Fraction of window to shift after each emission, defaults to 1/3
                        (e.g., 0.5 means 50% overlap between consecutive windows)
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source with a sequence of numbers
        data_source = SimpleDataProvider(range(10))

        # Create a linear scrubber with window size 4 and 50% overlap
        scrubber = LinearScrubber(window_length=4, shift_factor=0.5, source=data_source)

        # Process the windows
        for window in scrubber:
            print(f"Window values: {list(window.values)}")

        # Output:
        # Window values: [0, 1, 2, 3]
        # Window values: [2, 3, 4, 5]
        # Window values: [4, 5, 6, 7]
        # Window values: [6, 7, 8, 9]
        ```
    """

    def __init__(
        self, window_length: int = 100, shift_factor: float = 1.0 / 3.0, source: Handler[Any, T] | None = None
    ) -> None:
        """Initialize a linear scrubber with fixed window size and overlap.

        :param window_length: Number of points in each window, defaults to 100
        :param shift_factor: Fraction of window to shift after each emission, defaults to 1/3
        :param source: The handler providing input data, defaults to None
        """
        shift = max(1, int(shift_factor * window_length))
        super().__init__(take_condition=lambda buffer: len(buffer) >= window_length, shift=shift, source=source)
