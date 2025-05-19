from __future__ import annotations

from typing import Any

from pysatl_tsp.core.processor.inductive.moving_window_handler import MovingWindowHandler


class MidpointHandler(MovingWindowHandler[float | None, float | None]):
    """Midpoint price handler.

    Calculates the average of highest and lowest values over the period.
    This handler is useful for identifying the central price level within a range,
    providing a simple measure of the balance between high and low extremes.

    Inherits parameters from MovingWindowHandler:
    :param length: The period for the calculation, defaults to 10
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        # Create a midpoint handler with length of 4
        midpoint_handler = MidpointHandler(length=4)
        midpoint_handler.set_source(data_source)

        # Process the data
        for value in midpoint_handler:
            print(value)

        # Output:
        # None
        # None
        # None
        # 3.0  (midpoint of [1.0, 2.0, 3.0, 5.0] = (1.0 + 5.0) / 2 = 3.0)
        # 3.5  (midpoint of [2.0, 3.0, 5.0, 4.0] = (2.0 + 5.0) / 2 = 3.5)
        # 4.0  (midpoint of [3.0, 5.0, 4.0, 3.0] = (3.0 + 5.0) / 2 = 4.0)
        # 3.5  (midpoint of [5.0, 4.0, 3.0, 2.0] = (2.0 + 5.0) / 2 = 3.5)
        # 3.0  (midpoint of [4.0, 3.0, 2.0, 1.0] = (1.0 + 4.0) / 2 = 2.5)
        ```
    """

    def _compute_result(self, state: dict[str, Any]) -> float | None:
        """Calculate midpoint as (highest + lowest) / 2.

        Computes the average of the highest and lowest values in the current
        window. Returns None if there aren't enough values to fill the window
        or if any value in the window is None.

        :param state: Current state dictionary containing window values
        :return: Midpoint value or None if conditions aren't met
        """
        values = state["values"]

        if len(values) < self.length:
            return None

        highest = float("-inf")
        lowest = float("inf")
        for v in values:
            if v is None:
                return None
            highest = max(highest, v)
            lowest = min(lowest, v)

        return (highest + lowest) / 2
