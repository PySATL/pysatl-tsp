from collections.abc import Iterator
from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.processor.tee_handler import TeeHandler

from .ema_handler import EMAHandler


class DEMAHandler(Handler[float | None, float | None]):
    """A handler that calculates the Double Exponential Moving Average (DEMA).

    The Double Exponential Moving Average (DEMA) is designed to reduce the lag
    associated with traditional moving averages. It puts more weight on recent data
    by using the formula: DEMA = 2 * EMA - EMA of EMA.

    This implementation automatically configures a pipeline of EMA handlers to
    calculate both the primary EMA and the EMA of that EMA to produce DEMA values.

    :param length: The period for EMA calculations, defaults to 10
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create a DEMA handler with length of 3
        dema_handler = DEMAHandler(length=3)
        dema_handler.set_source(data_source)

        # Process the data
        for value in dema_handler:
            print(value)

        # The first few values will be None as the EMA needs to be established
        # Then the DEMA values will be calculated using 2 * EMA - EMA of EMA
        ```
    """

    def __init__(self, length: int = 10, source: Handler[Any, float | None] | None = None):
        """Initialize a DEMA handler.

        :param length: The period for EMA calculations, defaults to 10
        :param source: Input data source, defaults to None
        """
        self.length = length
        super().__init__(source)

    @staticmethod
    def _combine(ema: float | None, ema_of_ema: float | None) -> float | None:
        """Combine EMA and EMA-of-EMA to produce DEMA.

        Applies the formula: DEMA = 2 * EMA - EMA of EMA.
        If either input is None, returns None.

        :param ema: The primary EMA value
        :param ema_of_ema: The EMA of the EMA value
        :return: The calculated DEMA value or None if inputs are None
        """
        if ema is None or ema_of_ema is None:
            return None
        return 2 * ema - ema_of_ema

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields DEMA values.

        This method constructs a pipeline that:
        1. Takes values from the source
        2. Calculates the primary EMA
        3. Calculates the EMA of the EMA
        4. Combines them using the DEMA formula

        :return: Iterator yielding DEMA values
        :raises ValueError: If no source has been set
        """
        if not self.source:
            raise ValueError("Source is not set")

        yield from (
            self.source | EMAHandler(length=self.length) | TeeHandler(EMAHandler(length=self.length), self._combine)
        )
