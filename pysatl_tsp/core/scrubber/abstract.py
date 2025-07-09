from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Iterator
from itertools import islice
from typing import Any, Generic, overload

from pysatl_tsp.core import Handler, T

__all__ = ["Scrubber", "ScrubberWindow", "T"]


class ScrubberWindow(Generic[T]):
    """A sliding window container for time series data processing.

    ScrubberWindow provides a specialized container for holding a window of time series
    data values along with their corresponding indices. It's optimized for efficient
    append and remove operations at the ends of the window, making it suitable for
    sliding window algorithms in time series processing.

    This class manages two parallel deques: one for the actual data values and another
    for their corresponding indices or positions in the original data stream.

    :param values: Deque containing the data values, defaults to None (empty deque)
    :param indices: Deque containing the indices corresponding to values, defaults to None
                   (if not provided, sequential indices starting from 0 are used)
    :raises ValueError: If the lengths of values and indices don't match

    Example:
        ```python
        # Create an empty window
        window = ScrubberWindow()

        # Add values with automatic indices
        window.append(10.5)
        window.append(11.2)
        window.append(9.8)

        # Add value with explicit index
        window.append(12.1, index=100)

        # Get value by position in window
        first_value = window[0]  # 10.5

        # Get a slice of the window
        sub_window = window[1:3]  # Contains 11.2 and 9.8

        # Iterate through values
        for value in window:
            print(value)

        # Get original position of a value
        third_value_index = window.indices[2]  # 2
        fourth_value_index = window.indices[3]  # 100
        ```
    """

    def __init__(self, values: deque[T] | None = None, indices: deque[int] | None = None) -> None:
        """Initialize a scrubber window.

        :param values: Deque containing the data values, defaults to None (empty deque)
        :param indices: Deque containing the indices corresponding to values, defaults to None
                       (if not provided, sequential indices starting from 0 are used)
        :raises ValueError: If the lengths of values and indices don't match
        """
        if values is None:
            values = deque()
        if indices is None:
            indices = deque(range(len(values)))
        if len(values) != len(indices):
            raise ValueError("Values and indices of ScrubberWindow must be same length")
        self.values = values
        self.indices = indices

    def append(self, value: T, index: int | None = None) -> None:
        """Add a new value to the end of the window.

        :param value: The data value to append
        :param index: The index/position of the value in the original data stream,
                     defaults to None (auto-assigned as len(self))
        """
        self.values.append(value)
        if index is None:
            index = len(self)
        self.indices.append(index)

    def popleft(self) -> None:
        """Remove the oldest (leftmost) value from the window."""
        self.values.popleft()
        self.indices.popleft()

    def clear(self) -> None:
        """Remove all values from the window."""
        self.values.clear()
        self.indices.clear()

    def copy(self) -> ScrubberWindow[T]:
        """Create a deep copy of the window.

        :return: A new ScrubberWindow with copies of the values and indices
        """
        return ScrubberWindow(self.values.copy(), self.indices.copy())

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> ScrubberWindow[T]: ...

    def __getitem__(self, key: int | slice) -> T | ScrubberWindow[T]:
        """Get a value or sub-window by index or slice.

        :param key: Integer index or slice to retrieve
        :return: Single value (if key is int) or sub-window (if key is slice)
        :raises TypeError: If key is not an int or slice
        """
        match key:
            case int() as idx:
                return self.values[idx]

            case slice() as s:
                start, stop = s.start, s.stop
                if start and start < 0:
                    start = len(self) + start
                if stop and stop < 0:
                    stop = len(self) + stop
                return ScrubberWindow(
                    values=deque(islice(self.values, start, stop)),
                    indices=deque(islice(self.indices, start, stop)),
                )

            case _:
                raise TypeError(f"Unsupported key type: {type(key).__name__}")

    def __len__(self) -> int:
        """Get the number of values in the window.

        :return: The window size
        """
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        """Check if this window equals another window.

        :param other: Another object to compare with
        :return: True if other is a ScrubberWindow with equal values and indices
        """
        if not isinstance(other, ScrubberWindow):
            return NotImplemented
        return self.values == other.values and self.indices == other.indices

    def __hash__(self) -> int:
        """Get window's hash code.

        :return: The hash value of the object
        """
        return hash((self.values, self.indices))

    def __repr__(self) -> str:
        """Get a string representation of the window.

        :return: String representation showing values and indices
        """
        return f"ScrubberWindow(values: {self.values}, indices: {self.indices})"

    def __iter__(self) -> Iterator[T]:
        """Create an iterator over the values in the window.

        :return: Iterator yielding window values
        """
        return iter(self.values)


class Scrubber(Handler[T, ScrubberWindow[T]]):
    """Abstract base class for handlers that produce window views of time series data.

    Scrubbers consume individual data points and produce windows (sections) of the
    data stream. They are essential components for algorithms that need to analyze
    multiple data points together, such as moving averages, pattern detection,
    or feature extraction.

    Concrete implementations of this class define specific windowing strategies such
    as fixed-size sliding windows, tumbling windows, or context-based windows.

    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Example with a fixed-size sliding window scrubber (implementation not shown)

        # Create a data source
        data_source = SimpleDataProvider([10, 20, 30, 40, 50, 60, 70, 80])

        # Create a sliding window scrubber with window size 3
        window_scrubber = SlidingWindowScrubber(window_size=3, source=data_source)

        # Process windows
        for window in window_scrubber:
            # Each window is a ScrubberWindow instance
            print(f"Window values: {list(window.values)}")
            print(f"Window indices: {list(window.indices)}")

            # Calculate window statistics
            avg = sum(window.values) / len(window)
            print(f"Window average: {avg}")

        # Output:
        # Window values: [10, 20, 30]
        # Window indices: [0, 1, 2]
        # Window average: 20.0
        #
        # Window values: [20, 30, 40]
        # Window indices: [1, 2, 3]
        # Window average: 30.0
        #
        # ... and so on
        ```
    """

    def __init__(self, source: Handler[Any, T] | None = None) -> None:
        """Initialize a scrubber.

        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)

    @abstractmethod
    def __iter__(self) -> Iterator[ScrubberWindow[T]]:
        """Create an iterator that yields window views of the input data.

        Concrete implementations define specific windowing strategies.

        :return: Iterator yielding ScrubberWindow instances
        """
        pass
