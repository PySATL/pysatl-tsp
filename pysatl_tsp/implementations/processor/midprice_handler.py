from typing import Any

from pysatl_tsp.core.processor.inductive.moving_window_handler import MovingWindowHandler


class MidpriceHandler(MovingWindowHandler[tuple[float | None, float | None], float | None]):
    """Midprice handler that processes high/low price tuples.

    Calculates the average of the highest high and lowest low over a period.
    This handler is particularly useful for financial time series data where
    both high and low prices are available, providing a measure of the center
    of the price range over the specified period.

    Unlike simple averages, the midprice considers only extreme values, making
    it useful for range-bound markets and support/resistance identification.

    :param length: The period for the calculation, defaults to 10
    :param source: Input data source providing (high, low) tuples, defaults to None

    Example:
        ```python
        # Create a data source with (high, low) price tuples
        data = [
            (10.0, 8.0),  # (high, low)
            (11.0, 9.0),
            (12.0, 8.5),
            (10.5, 7.5),
            (11.5, 9.5),
            (13.0, 10.0),
        ]
        data_source = SimpleDataProvider(data)

        # Create a midprice handler with length of 3
        midprice_handler = MidpriceHandler(length=3)
        midprice_handler.set_source(data_source)

        # Process the data
        for value in midprice_handler:
            print(value)

        # Output:
        # None
        # None
        # 10.0  # (highest high 12.0 + lowest low 8.0) / 2 from first 3 tuples
        # 10.0  # (highest high 12.0 + lowest low 7.5) / 2
        # 10.0  # (highest high 12.0 + lowest low 7.5) / 2
        # 10.25 # (highest high 13.0 + lowest low 7.5) / 2
        ```
    """

    def _compute_result(self, state: dict[str, Any]) -> float | None:
        """Calculate midprice as (highest high + lowest low) / 2.

        Examines all high/low pairs in the current window, finds the highest high
        and lowest low values, and returns their average. This produces a measure
        of the central price level between the highest and lowest extremes.

        :param state: Current state dictionary containing window of (high, low) tuples
        :return: Midprice value or None if the window isn't filled or contains None values
        """
        values = state["values"]

        if len(values) < self.length:
            return None

        highest_high: float = float("-inf")
        lowest_low: float = float("inf")

        for high, low in values:
            if high is None or low is None:
                return None
            highest_high = max(highest_high, high)
            lowest_low = min(lowest_low, low)

        return (highest_high + lowest_low) / 2
