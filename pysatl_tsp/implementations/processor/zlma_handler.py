from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.processor.lag_handler import LagHandler
from pysatl_tsp.implementations.processor.ema_handler import EMAHandler


class ZLMAHandler(Handler[float | None, float | None]):
    """Zero Lag Moving Average (ZLMA) handler with lazy evaluation.

    Implements the formula ZLMA = MA(2 * close - close.shift(lag)), where lag = int(0.5 * (length - 1)).
    All calculations are performed lazily in a streaming fashion, computing values only
    when requested by the iterator.

    The ZLMA reduces lag by applying a forward-shifted moving average that compensates
    for the lag inherent in traditional moving averages.

    :param length: Period for the moving average calculation
    :param ma_handler: Moving average handler to apply. Default is EMA with the specified length.
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create a ZLMA handler with length of 5
        zlma_handler = ZLMAHandler(length=5)
        zlma_handler.set_source(data_source)

        # Process the data
        for value in zlma_handler:
            print(value)

        # Initial values will be None due to lag calculation requirements
        # Then ZLMA values will be calculated using the formula
        ```
    """

    def __init__(
        self,
        length: int = 10,
        ma_handler: Handler[Any, float | None] | None = None,
        source: Handler[Any, float | None] | None = None,
    ):
        """Initialize a Zero Lag Moving Average handler.

        :param length: Period for the moving average calculation, defaults to 10
        :param ma_handler: Moving average handler to apply, defaults to EMAHandler with the specified length
        :param source: Input data source, defaults to None
        """
        super().__init__(source=source)
        self.length = length
        self.ma_handler = ma_handler if ma_handler is not None else EMAHandler(length=length)

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields ZLMA values.

        This method implements the ZLMA calculation pipeline according to the formula:
        ZLMA = MA(2 * close - close.shift(lag)), where lag = int(0.5 * (length - 1)).

        :return: Iterator yielding ZLMA values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("ZLMAHandler requires a data source")

        # Calculate lag
        lag = int(0.5 * (self.length - 1))

        yield from self.source | LagHandler(lag=lag) | self.ma_handler
