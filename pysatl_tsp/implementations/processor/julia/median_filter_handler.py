from typing import Any

from pysatl_tsp.core import Handler

from ._base import JuliaHandler, JuliaOpSpec


class JuliaMedianFilterHandler(JuliaHandler):
    """Median filter Handler backed by a Julia implementation.

    Replaces every observation with the median of a sliding window ending at
    it, which suppresses impulse noise and outliers while preserving edges far
    better than an averaging filter does.

    When this handler directly follows another Julia-backed handler, both are
    evaluated inside a single Julia chain, so the pipeline crosses the language
    boundary once per observation regardless of how many Julia handlers it
    contains.

    None values are ignored in the calculation, and a minimum number of valid
    observations can be required before producing a result.

    The Julia runtime is initialized lazily on the first computed value, so
    constructing the handler (and building pipelines with it) is cheap and does
    not require Julia at all.

    :param length: The window size for the median calculation, defaults to 10
    :param min_periods: Minimum number of observations required to have a value, defaults to length
    :param source: Input data source, defaults to None

    Example:
        ```python
        data_source = SimpleDataProvider([1.0, 2.0, 100.0, 3.0, 4.0])

        # Smooth the series first, then drop the remaining spikes;
        # both steps are executed by a single Julia chain
        pipeline = data_source | JuliaSMAHandler(length=3) | JuliaMedianFilterHandler(length=3)

        for value in pipeline:
            print(value)
        ```
    """

    def __init__(
        self, length: int = 10, min_periods: int | None = None, source: Handler[Any, float | None] | None = None
    ):
        """Initialize Julia-backed median filter with specified parameters.

        :param length: The window size for the median calculation, defaults to 10
        :param min_periods: Minimum number of non-None observations required to calculate a result,
                           defaults to length if None
        :param source: Input data source, defaults to None
        """
        super().__init__(source)
        self.length = length if length and length > 0 else 10
        self.min_periods = min_periods if min_periods is not None else self.length

    def _julia_op_spec(self) -> JuliaOpSpec:
        """Describe this handler as a Julia median filter operation.

        :return: Specification of the Julia ``median_filter`` operation
        """
        return JuliaOpSpec("median_filter", (self.length, self.min_periods))
