from __future__ import annotations

from collections.abc import Iterator
from itertools import tee, zip_longest
from typing import Any

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.core.handler import Handler
from pysatl_tsp.implementations.processor.ema_handler import EMAHandler


class T3Handler(Handler[float | None, float | None]):
    """Tim Tillson's T3 Moving Average handler with lazy evaluation.

    This handler implements the T3 adaptive moving average developed by Tim Tillson,
    which is designed to reduce lag and improve smoothness. The T3 is calculated as:
    T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3, where e1-e6 are sequentially computed EMAs.

    All calculations are performed lazily in a streaming fashion, computing values only
    when requested by the iterator.

    :param length: Period for each EMA calculation, defaults to 10
    :param a: Volume factor (0 < a < 1), controls smoothness vs. responsiveness, defaults to 0.7
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create a T3 handler with length of 5 and volume factor of 0.7
        t3_handler = T3Handler(length=5, a=0.7)
        t3_handler.set_source(data_source)

        # Process the data
        for value in t3_handler:
            print(value)

        # Initial values will be None as the T3 calculation requires
        # six sequential EMA calculations to establish
        ```
    """

    def __init__(self, length: int = 10, a: float = 0.7, source: Handler[Any, float | None] | None = None):
        """Initialize a T3 moving average handler.

        :param length: Period for each EMA calculation, defaults to 10
        :param a: Volume factor (0 < a < 1), controls smoothness vs. responsiveness, defaults to 0.7
        :param source: Input data source, defaults to None
        """
        super().__init__(source=source)
        self.length = length
        self.a = a if 0 < a < 1 else 0.7

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields T3 moving average values.

        This method constructs a pipeline of six cascaded EMA calculations, where
        each EMA takes the output of the previous one as input. The final T3 value
        is a weighted sum of the last four EMAs in the sequence.

        :return: Iterator yielding T3 values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("T3Handler requires a data source")

        # Calculate coefficients
        c1 = -self.a * self.a**2
        c2 = 3 * self.a**2 + 3 * self.a**3
        c3 = -6 * self.a**2 - 3 * self.a - 3 * self.a**3
        c4 = self.a**3 + 3 * self.a**2 + 3 * self.a + 1

        # Calculate e1 (first EMA)
        e1_provider = SimpleDataProvider(self.source)
        e1_pipeline = e1_provider | EMAHandler(length=self.length)

        # Calculate e2 (second EMA based on e1)
        e2_provider = SimpleDataProvider(iter(e1_pipeline))
        e2_pipeline = e2_provider | EMAHandler(length=self.length)

        # Calculate e3 (third EMA based on e2)
        e3_provider = SimpleDataProvider(iter(e2_pipeline))
        e3_pipeline = e3_provider | EMAHandler(length=self.length)
        e3_iter = iter(e3_pipeline)

        # Create copies of e3 for use in T3 formula and for calculating e4
        e3_for_t3, e3_for_e4 = tee(e3_iter)

        # Calculate e4 (fourth EMA based on e3)
        e4_provider = SimpleDataProvider(e3_for_e4)
        e4_pipeline = e4_provider | EMAHandler(length=self.length)
        e4_iter = iter(e4_pipeline)

        # Create copies of e4 for use in T3 formula and for calculating e5
        e4_for_t3, e4_for_e5 = tee(e4_iter)

        # Calculate e5 (fifth EMA based on e4)
        e5_provider = SimpleDataProvider(e4_for_e5)
        e5_pipeline = e5_provider | EMAHandler(length=self.length)
        e5_iter = iter(e5_pipeline)

        # Create copies of e5 for use in T3 formula and for calculating e6
        e5_for_t3, e5_for_e6 = tee(e5_iter)

        # Calculate e6 (sixth EMA based on e5)
        e6_provider = SimpleDataProvider(e5_for_e6)
        e6_pipeline = e6_provider | EMAHandler(length=self.length)
        e6_iter = iter(e6_pipeline)

        # Combine e3, e4, e5, e6 to calculate T3
        for e3, e4, e5, e6 in zip_longest(e3_for_t3, e4_for_t3, e5_for_t3, e6_iter):
            if e3 is None or e4 is None or e5 is None or e6 is None:
                yield None
            else:
                t3_value = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
                yield t3_value
