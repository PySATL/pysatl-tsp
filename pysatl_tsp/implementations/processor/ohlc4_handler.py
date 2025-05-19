from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.processor.mapping_handler import MappingHandler


class Ohlc4Handler(MappingHandler[tuple[float | None, float | None, float | None, float | None], float | None]):
    """A handler that calculates the average of OHLC (Open, High, Low, Close) price data.

    This handler computes the OHLC4 (also known as the typical price), which is the simple
    arithmetic mean of the Open, High, Low, and Close prices for each time period.
    The calculation helps smooth price data and can be used as input for other indicators.

    :param source: The handler providing OHLC tuples, defaults to None

    Example:
        ```python
        # Create a data source with OHLC price tuples
        ohlc_data = [
            (100.0, 105.0, 98.0, 103.0),  # (open, high, low, close)
            (103.0, 107.0, 101.0, 104.0),
            (104.0, 109.0, 102.0, 108.0),
        ]
        data_source = SimpleDataProvider(ohlc_data)

        # Create an OHLC4 handler
        ohlc4_handler = Ohlc4Handler(source=data_source)

        # Process the data
        for value in ohlc4_handler:
            print(value)

        # Output:
        # 101.5  # (100.0 + 105.0 + 98.0 + 103.0) / 4
        # 103.75 # (103.0 + 107.0 + 101.0 + 104.0) / 4
        # 105.75 # (104.0 + 109.0 + 102.0 + 108.0) / 4
        ```
    """

    @staticmethod
    def _map_func(t: tuple[float | None, float | None, float | None, float | None]) -> float | None:
        """Calculate the average of OHLC values.

        Takes a tuple of Open, High, Low, Close values and returns their arithmetic mean.
        If any value in the tuple is None, returns None.

        :param t: Tuple of (open, high, low, close) values
        :return: Average of OHLC values or None if any value is None
        """
        res = 0.0
        for x in t:
            if x is None:
                return None
            res += x
        return res / 4

    def __init__(
        self, source: Handler[Any, tuple[float | None, float | None, float | None, float | None]] | None = None
    ):
        """Initialize an OHLC4 handler.

        :param source: The handler providing OHLC tuples, defaults to None
        """
        super().__init__(self._map_func, source)
