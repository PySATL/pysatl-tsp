from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from pysatl_tsp.core import Handler


class LagHandler(Handler[float | None, float | None]):
    """A handler that applies a lag-based transformation to time series data.

    This handler applies a formula (2 * current_value - lagged_value) that compares
    the current value with a value from 'lag' time steps in the past. For the first
    'lag' values where no lagged value is available, it outputs None.

    The transformation can be useful for detecting changes or trends in time series
    by comparing current values with historical ones.

    :param lag: Number of time steps to look back for the lagged value

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0])

        # Create a lag handler with lag of 2
        lag_handler = LagHandler(lag=2)
        lag_handler.set_source(data_source)

        # Process the data
        results = list(lag_handler)
        print(results)

        # Output:
        # [None, None, 5.0, 6.0, 7.0]
        #
        # Explanation:
        # - First two values: None (not enough history)
        # - Third value: 2*3.0-1.0 = 5.0 (current=3.0, lagged=1.0)
        # - Fourth value: 2*4.0-2.0 = 6.0 (current=4.0, lagged=2.0)
        # - Fifth value: 2*5.0-3.0 = 7.0 (current=5.0, lagged=3.0)
        ```
    """

    def __init__(self, lag: int):
        """Initialize a lag handler.

        :param lag: Number of time steps to look back for the lagged value
        """
        super().__init__()
        self.lag = lag

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields transformed values based on lag comparison.

        This method outputs None for the first 'lag' values, then applies the formula
        2 * current_value - lagged_value for subsequent values. If either the current
        value or the lagged value is None, the result will be None.

        :return: Iterator yielding transformed values or None
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("LagHandler requires a data source")

        source_iter = iter(self.source)
        buffer: deque[float | None] = deque(maxlen=self.lag + 1)

        try:
            for _ in range(self.lag):
                buffer.append(next(source_iter))
                yield None  # First 'lag' values have no result
        except StopIteration:
            return

        # Apply formula for each new value
        for current_value in source_iter:
            buffer.append(current_value)  # Add new value
            lagged_value = buffer.popleft()

            if current_value is None or lagged_value is None:
                yield None
            else:
                yield 2 * current_value - lagged_value
