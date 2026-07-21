from typing import Any

from pysatl_tsp.core import Handler

from ._base import JuliaHandler, JuliaOpSpec


class JuliaSMAHandler(JuliaHandler):
    """Simple Moving Average (SMA) Handler backed by a Julia implementation.

    Behaves identically to :class:`SMAHandler` and is a drop-in replacement for
    it inside processing pipelines, but delegates the computation to the Julia
    backend. This makes it possible to move performance-critical numerical code
    to Julia while keeping the library's Handler/Pipeline architecture
    unchanged.

    When this handler directly follows another Julia-backed handler, both are
    evaluated inside a single Julia chain, so the pipeline crosses the language
    boundary once per observation regardless of how many Julia handlers it
    contains.

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
        super().__init__(source)
        self.length = length if length and length > 0 else 10
        self.min_periods = min_periods if min_periods is not None else self.length

    def _julia_op_spec(self) -> JuliaOpSpec:
        """Describe this handler as a Julia simple moving average operation.

        :return: Specification of the Julia ``sma`` operation
        """
        return JuliaOpSpec("sma", (self.length, self.min_periods))
