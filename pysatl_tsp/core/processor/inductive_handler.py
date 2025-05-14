from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from pysatl_tsp.core import Handler, T, U


class InductiveHandler(Handler[T, U], ABC):
    """A base class for implementing inductive time series processing algorithms.

    An inductive function is one where the value for a sequence x[1]...x[n]
    can be computed from its value for the sequence x[1]...x[n-1] and the new element x[n].
    Mathematically, there exists a function F such that:
    f(⟨x[1], ..., x[n]⟩) = F( f(⟨x[1], ..., x[n-1]⟩), x[n])

    This class provides a structured approach to implementing such algorithms by
    separating the processing into three key phases:
    1. Initializing the initial state
    2. Updating the state with each new value
    3. Computing the result based on the current state

    Common examples of inductive functions include moving averages, exponential
    moving averages (EMA), cumulative sums, and Welford's algorithm for variance calculation.

    :param source: The handler providing input data, defaults to None
    """

    def __init__(self, source: Handler[Any, T] | None = None):
        """Initialize an inductive handler.

        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)

    @abstractmethod
    def _initialize_state(self) -> Any:
        """Initialize the starting state for the inductive algorithm.

        This method is called once when the iterator first starts and should create
        a data structure representing the initial state of the algorithm.

        :return: An object of arbitrary type representing the initial state

        Example:
            ```python
            # For a simple moving average:
            def _initialize_state(self):
                return {"values": [], "sum": 0.0}


            # For an EMA:
            def _initialize_state(self):
                return {"ema": None, "values": []}


            # For standard deviation (Welford's algorithm):
            def _initialize_state(self):
                return {"count": 0, "mean": 0.0, "M2": 0.0}
            ```
        """
        pass

    @abstractmethod
    def _update_state(self, state: Any, value: T) -> Any:
        """Update the handler's state with a new input value.

        This method embodies the core logic of the inductive function: how the current
        state transforms when a new value arrives.

        :param state: The current state of the handler, created by _initialize_state
                      or returned by the previous call to _update_state
        :param value: The new value from the data source
        :return: The updated state of the same type as the state parameter

        Example:
            ```python
            # For a simple fixed-window moving average:
            def _update_state(self, state, value):
                state["values"].append(value)
                state["sum"] += value
                if len(state["values"]) > self.length:
                    state["sum"] -= state["values"].pop(0)
                return state


            # For an EMA:
            def _update_state(self, state, value):
                if state["ema"] is None:
                    state["values"].append(value)
                    if len(state["values"]) >= self.length:
                        state["ema"] = sum(state["values"]) / len(state["values"])
                        state["values"] = []
                else:
                    state["ema"] = self.alpha * value + (1 - self.alpha) * state["ema"]
                return state
            ```
        """
        pass

    @abstractmethod
    def _compute_result(self, state: Any) -> U:
        """Compute the output result based on the current state.

        Transforms the internal state of the handler into an output value that
        will be passed to the next handler in the chain or returned to the end user.

        :param state: The current state of the handler
        :return: The processing result corresponding to the current state

        Example:
            ```python
            # For a simple moving average:
            def _compute_result(self, state):
                if not state["values"]:
                    return float("nan")
                return state["sum"] / len(state["values"])


            # For standard deviation:
            def _compute_result(self, state):
                if state["count"] < 2:
                    return 0.0
                return math.sqrt(state["M2"] / state["count"])
            ```
        """
        pass

    def __iter__(self) -> Iterator[U]:
        """Create an iterator that sequentially computes results for each input value.

        This method implements the main processing loop of the inductive handler:
        1. Initialize the starting state
        2. For each value from the source:
           - Update the state with the new value
           - Compute the result based on the updated state
           - Yield the result as the next value of the iterator

        :return: An iterator yielding processed results for each input value
        :raises ValueError: If no source has been set

        Note:
            The iterator resets the internal state each time it's called,
            allowing the same handler to be reused for multiple iterations.
        """
        if self.source is None:
            return

        self._state = self._initialize_state()
        for value in self.source:
            self._state = self._update_state(self._state, value)
            next_value = self._compute_result(self._state)
            if next_value:
                yield next_value
