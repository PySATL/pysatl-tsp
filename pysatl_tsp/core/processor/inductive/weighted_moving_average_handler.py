from abc import ABC, abstractmethod
from typing import Any

from pysatl_tsp.core import Handler

from .moving_window_handler import MovingWindowHandler


class WeightedMovingAverageHandler(MovingWindowHandler[float | None, float | None], ABC):
    """
    Abstract base class for weighted moving average handlers.

    This class provides common functionality for different types of weighted
    moving averages, such as WMA (linearly weighted) and FWMA (Fibonacci weighted).

    :param length: The period for the moving average calculation, defaults to 10
    :param asc: Whether weights should be in ascending order, defaults to True
    :param source: Input data source, defaults to None
    """

    def __init__(self, length: int = 10, asc: bool = True, source: Handler[Any, float | None] | None = None):
        """Initialize weighted moving average handler with specified parameters."""
        super().__init__(length=length, source=source)
        self.length = length if length and length > 0 else 10
        self.asc = asc if asc is not None else True
        # Calculate the weights - this calls the abstract method that subclasses must implement
        self.weights = self._calculate_weights(self.length, self.asc)

    @abstractmethod
    def _calculate_weights(self, length: int, asc: bool) -> list[float]:
        """
        Calculate the weights for the moving average.

        This method should be implemented by subclasses to provide
        specific weighting schemes (linear, Fibonacci, etc.).

        :param length: The number of weights to generate
        :param asc: Whether weights should be in ascending order
        :return: A list of normalized weights that sum to 1.0
        """
        pass

    def _compute_result(self, state: dict[str, Any]) -> float | None:
        """
        Calculate weighted average based on current values and weights.

        :param state: Current state containing the values
        :return: Weighted average or None if not enough data or any value is None
        """
        values = state["values"]

        if len(values) < self.length:
            return None

        weighted_sum: float = 0
        for v, w in zip(values, self.weights):
            if v is None:
                return None
            weighted_sum += v * w

        return weighted_sum
