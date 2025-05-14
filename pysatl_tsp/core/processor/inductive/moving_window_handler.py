from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.handler import T, U
from pysatl_tsp.core.processor.inductive.inductive_handler import InductiveHandler


class MovingWindowHandler(InductiveHandler[T, U], ABC):
    """Abstract base class for handlers that operate on a moving window of values.

    This base class manages a window of values and provides the framework for
    different types of moving window calculations.

    :param length: The period for the calculation, defaults to 10
    :param source: Input data source, defaults to None
    """

    def __init__(self, length: int = 10, source: Handler[Any, T] | None = None):
        """Initialize moving window handler with specified parameters.

        :param length: The period for the calculation, defaults to 10
        :param source: Input data source, defaults to None
        """
        super().__init__(source)
        self.length = length if length and length > 0 else 10

    def _initialize_state(self) -> dict[str, Any]:
        """Initialize state for moving window calculation.

        Creates an empty deque with maximum length set to the window length.

        :return: Dictionary containing initial state with empty values queue
        """
        return {
            "values": deque(maxlen=self.length),
        }

    def _update_state(self, state: dict[str, Any], value: T) -> dict[str, Any]:
        """Update state with a new value.

        Maintains a rolling window of the last 'length' values by appending
        the new value to the deque. If the deque is full, the oldest value
        is automatically removed.

        :param state: Current state dictionary
        :param value: New value to add to the window
        :return: Updated state dictionary
        """
        state["values"].append(value)
        return state

    @abstractmethod
    def _compute_result(self, state: dict[str, Any]) -> U:
        """Calculate result based on current values in the window.

        This abstract method must be implemented by subclasses to define
        how to compute a result from the current window of values.

        :param state: Current state dictionary containing window of values
        :return: Computed result from the window values
        """
        pass
