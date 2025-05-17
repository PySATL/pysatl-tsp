from __future__ import annotations

from collections.abc import Iterator
from itertools import tee, zip_longest

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.core.handler import Handler
from pysatl_tsp.implementations.processor.ema_handler import EMAHandler


class TEMAHandler(Handler[float | None, float | None]):
    """Triple Exponential Moving Average (TEMA) handler with lazy evaluation.

    This handler implements the TEMA indicator developed by Patrick Mulloy, which
    reduces lag by applying the formula: TEMA = 3 * (EMA1 - EMA2) + EMA3.
    Here EMA1 is the EMA of the original data, EMA2 is the EMA of EMA1, and
    EMA3 is the EMA of EMA2.

    All calculations are performed lazily in a streaming fashion, computing values only
    when requested by the iterator.

    :param length: Period for each EMA calculation

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create a TEMA handler with length of 5
        tema_handler = TEMAHandler(length=5)
        tema_handler.set_source(data_source)

        # Process the data
        for value in tema_handler:
            print(value)

        # Initial values will be None as TEMA requires three levels of EMA
        # After initialization, TEMA values will follow the price action more closely
        # than a regular EMA while maintaining smoothness
        ```
    """

    def __init__(self, length: int):
        """Initialize a TEMA handler.

        :param length: Period for each EMA calculation
        """
        super().__init__()
        self.length = length

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields TEMA values.

        This method constructs a pipeline of three cascaded EMA calculations and
        applies the TEMA formula: 3 * (EMA1 - EMA2) + EMA3.

        :return: Iterator yielding TEMA values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("TEMAHandler requires a data source")

        # Calculate EMA1
        ema1_provider = SimpleDataProvider(self.source)
        ema1_pipeline = ema1_provider | EMAHandler(length=self.length)

        # Create two copies of the EMA1 iterator
        ema1_iter1, ema1_iter2 = tee(iter(ema1_pipeline))

        # Calculate EMA2 based on the first copy of EMA1
        ema2_provider = SimpleDataProvider(ema1_iter1)
        ema2_pipeline = ema2_provider | EMAHandler(length=self.length)

        # Create two copies of the EMA2 iterator
        ema2_iter1, ema2_iter2 = tee(iter(ema2_pipeline))

        # Calculate EMA3 based on the first copy of EMA2
        ema3_provider = SimpleDataProvider(ema2_iter1)
        ema3_pipeline = ema3_provider | EMAHandler(length=self.length)
        ema3_iter = iter(ema3_pipeline)

        # Combine all three iterators and apply the TEMA formula
        for ema1, ema2, ema3 in zip_longest(ema1_iter2, ema2_iter2, ema3_iter):
            if ema1 is None or ema2 is None or ema3 is None:
                yield None
            else:
                tema_value = 3 * (ema1 - ema2) + ema3
                yield tema_value
