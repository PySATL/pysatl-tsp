from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.processor.inductive.inductive_handler import InductiveHandler


class RMAHandler(InductiveHandler[float | None, float | None]):
    """Wilder's Moving Average (RMA) Handler.

    The Wilder's Moving Average is an Exponential Moving Average (EMA) with
    a modified alpha = 1 / length. It was introduced by J. Welles Wilder and is
    also known as the Smoothed Moving Average.

    RMA gives greater weight to recent data and less weight to older data, but
    does so more gradually than a standard EMA, resulting in a smoother line.
    It's commonly used in technical analysis for calculating indicators like
    the Relative Strength Index (RSI).

    :param length: The number of periods for the moving average calculation, defaults to 10
    :param source: Source handler providing the input data, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create an RMA handler with length of 5
        rma_handler = RMAHandler(length=5)
        rma_handler.set_source(data_source)

        # Process the data
        for value in rma_handler:
            print(value)

        # First values will be None until we have 'length' values
        # Then RMA values will be calculated with alpha = 1/5
        ```
    """

    def __init__(self, length: int = 10, source: Handler[Any, float | None] | None = None):
        """Initialize a Wilder's Moving Average handler.

        :param length: The number of periods for the moving average calculation, defaults to 10
        :param source: Source handler providing the input data, defaults to None
        """
        super().__init__(source)
        self.length = length
        self.alpha = 1 / length

    def _initialize_state(self) -> dict[str, float | int]:
        """Initialize state for RMA calculation.

        Creates an initial state dictionary with counters and accumulators set to zero.

        :return: Dictionary containing initial state variables
        """
        return {"enumerator": 0, "denominator": 0, "not_none_count": 0}

    def _update_state(self, state: dict[str, float | int], value: float | None) -> dict[str, float | int]:
        """Update state with a new value.

        Updates the running totals using the RMA formula with alpha = 1/length.
        Handles None values by treating them as zeros in the calculation but
        tracking how many non-None values have been seen.

        :param state: Current state dictionary
        :param value: New value to incorporate into the RMA calculation
        :return: Updated state dictionary
        """
        state["not_none_count"] += int(value is not None)
        state["denominator"] = (1 - self.alpha) * state["denominator"] + int(value is not None)
        if value is None:
            value = 0

        state["enumerator"] = (1 - self.alpha) * state["enumerator"] + value
        return state

    def _compute_result(self, state: dict[str, float | int]) -> float | None:
        """Calculate the RMA value based on current state.

        Returns the current RMA value if there's a non-zero denominator and we have
        seen at least 'length' non-None values. Otherwise returns None.

        :param state: Current state dictionary
        :return: Current RMA value or None if conditions aren't met
        """
        if not state["denominator"] or state["not_none_count"] < self.length:
            return None
        return state["enumerator"] / state["denominator"]
