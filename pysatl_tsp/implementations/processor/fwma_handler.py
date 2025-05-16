from pysatl_tsp.core.processor.inductive.weighted_moving_average_handler import WeightedMovingAverageHandler


class FWMAHandler(WeightedMovingAverageHandler):
    """Fibonacci Weighted Moving Average (FWMA) handler.

    Calculates a moving average using Fibonacci sequence numbers as weights.
    The Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, ...) provides a natural
    weighting scheme where each number is the sum of the two preceding ones.

    By default, higher weights are assigned to more recent values (when asc=False),
    making this moving average more responsive to recent changes in the data.

    Inherits general behavior from WeightedMovingAverageHandler, only changing
    the weight calculation method.

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # Create a FWMA handler with length of 5
        fwma_handler = FWMAHandler(length=5)
        fwma_handler.set_source(data_source)

        # Process the data
        for value in fwma_handler:
            print(value)

        # First 4 values will be None (not enough data points)
        # Subsequent values will be weighted averages using Fibonacci weights
        # For length=5, weights would be [0.01, 0.01, 0.02, 0.03, 0.05] (normalized)
        # or [0.05, 0.03, 0.02, 0.01, 0.01] when asc=False (default)
        ```
    """

    def _calculate_weights(self, length: int, asc: bool) -> list[float]:
        """Calculate Fibonacci weights for FWMA.

        Generates weights based on the Fibonacci sequence and normalizes them
        to sum to 1.0. The sequence order can be reversed based on the asc parameter.

        :param length: The number of weights to generate
        :param asc: Whether weights should be in ascending order
        :return: A list of normalized weights summing to 1.0
        """
        sequence = self._fibonacci_sequence(length)

        if not asc:
            sequence = sequence[::-1]

        # Normalize the weights to sum to 1.0
        total = sum(sequence)
        return [x / total for x in sequence]

    def _fibonacci_sequence(self, n: int) -> list[float]:
        """Generate the Fibonacci sequence of specified length.

        Creates a list containing the first n numbers in the Fibonacci sequence.
        The sequence starts with 1, 1 and each subsequent number is the sum of
        the two preceding ones.

        :param n: The length of the sequence to generate
        :return: A list containing the Fibonacci sequence
        """
        if n <= 0:
            return []

        if n == 1:
            return [1.0]

        sequence = [1.0, 1.0]
        for i in range(2, n):
            sequence.append(sequence[i - 1] + sequence[i - 2])

        return sequence
