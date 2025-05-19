import numpy as np

from pysatl_tsp.core.processor.inductive.weighted_moving_average_handler import WeightedMovingAverageHandler


class WMAHandler(WeightedMovingAverageHandler):
    """Weighted Moving Average (WMA) handler.

    Calculates a moving average where each data point is multiplied by a weight
    before being included in the average. The weights increase or decrease linearly,
    giving more importance to recent data points by default.

    This implementation matches the functionality of pandas_ta.wma, using a linear
    weighting scheme where the weight of each value is proportional to its position
    in the window.

    Inherits parameters from WeightedMovingAverageHandler:
    :param length: The period for the calculation, defaults to 10
    :param asc: Whether weights should be applied in ascending order, defaults to False
                When False, most recent values get higher weights
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # Create a WMA handler with length of 4
        wma_handler = WMAHandler(length=4)
        wma_handler.set_source(data_source)

        # Process the data
        for value in wma_handler:
            print(value)

        # First 3 values will be None (not enough data points)
        # For length=4, weights would be [0.1, 0.2, 0.3, 0.4] (with asc=False)
        # So the 4th value would be: (1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 4.0*0.4) = 3.0
        # Similarly, the 5th value: (2.0*0.1 + 3.0*0.2 + 4.0*0.3 + 5.0*0.4) = 4.0
        ```
    """

    def _calculate_weights(self, length: int, asc: bool) -> list[float]:
        """Calculate linear weights for WMA.

        Generates a sequence of linearly increasing weights that sum to 1.0.
        The weights increase from 1 to length, and are then normalized by dividing
        by their sum. The order can be reversed based on the asc parameter.

        :param length: The number of weights to generate
        :param asc: Whether weights should be in ascending order
                   When False (default), higher weights are assigned to more recent values
        :return: A list of normalized weights summing to 1.0
        """
        weights: list[float] = list(np.arange(1, length + 1))
        if not asc:
            weights = weights[::-1]

        # Calculate total weight (sum of weights)
        total_weight = np.sum(weights)

        # Return normalized weights
        return [float(w) / total_weight for w in weights]
