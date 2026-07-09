from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.processor.inductive.moving_window_handler import MovingWindowHandler

from ._bridge import get_moving_average_module


class JuliaSMAHandler(MovingWindowHandler[float | None, float | None]):
    """Simple Moving Average (SMA) Handler backed by a Julia implementation.

    Behaves identically to :class:`SMAHandler` and is a drop-in replacement for
    it inside processing pipelines, but delegates the window computation to the
    Julia ``MovingAverage.sma`` function via juliacall. This makes it possible
    to move performance-critical numerical code to Julia while keeping the
    library's Handler/Pipeline architecture unchanged.

    None values are ignored in the calculation, and a minimum number of valid
    observations can be required before producing a result.

    The Julia runtime is initialized lazily on the first computed value, so
    constructing the handler (and building pipelines with it) is cheap and does
    not require Julia at all.

    :param length: The period for the SMA calculation, defaults to 10
    :param min_periods: Minimum number of observations required to have a value, defaults to length
    :param source: Input data source, defaults to None

    Example:
        ```python
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0])

        pipeline = data_source | JuliaSMAHandler(length=4, min_periods=3)

        for value in pipeline:
            print(value)
        ```
    """

    def __init__(
        self, length: int = 10, min_periods: int | None = None, source: Handler[Any, float | None] | None = None
    ):
        """Initialize Julia-backed SMA handler with specified parameters.

        :param length: The period for the SMA calculation, defaults to 10
        :param min_periods: Minimum number of non-None observations required to calculate a result,
                           defaults to length if None
        :param source: Input data source, defaults to None
        """
        super().__init__(length=length, source=source)
        self.min_periods = min_periods if min_periods is not None else length
        self._backend: Any | None = None

    def _compute_result(self, state: dict[str, Any]) -> float | None:
        """Calculate simple moving average by delegating to the Julia backend.

        The current window is passed to Julia as a vector where Python None is
        converted to Julia `nothing`. The backend returns either a Float64 or
        `nothing`, which is converted back to float or None respectively.

        :param state: Current state containing the values in the moving window
        :return: Simple moving average of valid values or None if there aren't enough valid values
        """
        if self._backend is None:
            self._backend = get_moving_average_module()

        result = self._backend.sma(list(state["values"]), self.min_periods)
        return None if result is None else float(result)
