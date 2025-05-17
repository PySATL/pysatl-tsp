import math

from pysatl_tsp.core.processor.inductive.weighted_moving_average_handler import WeightedMovingAverageHandler


class PWMAHandler(WeightedMovingAverageHandler):
    """Pascal Weighted Moving Average (PWMA) handler.

    Calculates a weighted moving average using coefficients from Pascal's triangle
    as weights. Pascal's triangle provides a natural weighting scheme where the
    central values receive more weight than the extremes, creating a balanced
    but still centered weighting distribution.

    This implementation matches the functionality of pandas_ta.pwma.

    Inherits parameters from WeightedMovingAverageHandler:
    :param length: The period for the calculation, defaults to 10
    :param asc: Whether weights should be applied in ascending order, defaults to False
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # Create a PWMA handler with length of 4
        pwma_handler = PWMAHandler(length=4)
        pwma_handler.set_source(data_source)

        # Process the data
        for value in pwma_handler:
            print(value)

        # First 3 values will be None (not enough data points)
        # For length=4, Pascal weights would be [1/8, 3/8, 3/8, 1/8] or [1/8, 3/8, 3/8, 1/8] when asc=False
        # The calculation gives more weight to the central values
        ```
    """

    def _calculate_weights(self, length: int, asc: bool) -> list[float]:
        """Calculate Pascal's triangle weights for PWMA.

        Generates weights based on a row of Pascal's triangle and normalizes them
        to sum to 1.0. The row is determined by (length-1) to match the pandas_ta
        implementation. The weights order can be reversed based on the asc parameter.

        :param length: The number of weights to generate
        :param asc: Whether weights should be in ascending order
        :return: A list of normalized weights summing to 1.0
        """
        # Generate the row of Pascal's triangle
        # Using (length-1) to match pandas_ta implementation
        triangle = [self._combination(length - 1, i) for i in range(length)]

        # Normalize the weights to sum to 1.0
        total = sum(triangle)
        weights = [w / total for w in triangle]

        if not asc:
            weights = weights[::-1]

        return weights

    def _combination(self, n: int, r: int) -> float:
        """Calculate the binomial coefficient (n choose r).

        Computes the binomial coefficient, also known as "n choose r", which
        represents the number of ways to choose r items from a set of n items
        without regard to order.

        Uses math.comb when available (Python 3.8+) for efficiency, otherwise
        falls back to calculation using factorials.

        :param n: The total number of items
        :param r: The number of items to choose
        :return: The binomial coefficient
        """
        return math.comb(n, r)

    def _factorial(self, n: int) -> int:
        """Calculate the factorial of n.

        Computes n! recursively. This method is used as a fallback when
        math.comb is not available for binomial coefficient calculation.

        :param n: The number to calculate factorial for
        :return: The factorial of n
        """
        if n <= 1:
            return 1
        return n * self._factorial(n - 1)
