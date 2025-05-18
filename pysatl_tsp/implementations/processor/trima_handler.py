from collections.abc import Iterator
from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.implementations.processor.sma_handler import SMAHandler


class TRIMAHandler(Handler[float | None, float | None]):
    """Triangular Moving Average (TRIMA) Handler.

    A weighted moving average where the shape of the weights are triangular and the
    greatest weight is in the middle of the period. Implemented as a pipeline of two
    SMA calculations with half of the requested length.

    TRIMA gives more weight to the middle portion of the price series and less weight
    to the oldest and newest data. It is slower to respond to price changes but better
    at filtering out market noise than a simple moving average.

    :param length: The period for the TRIMA calculation, defaults to 10
    :param source: Input data source, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create a TRIMA handler with length of 5
        trima_handler = TRIMAHandler(length=5)
        trima_handler.set_source(data_source)

        # Process the data
        for value in trima_handler:
            print(value)

        # For a TRIMA with length=5, half_length=3:
        # First SMA(3) requires 3 values
        # Second SMA(3) of the first SMA values requires another 3 values
        # So the first few values will be None, then TRIMA values follow
        # The TRIMA gives more weight to the middle values in the calculation
        ```
    """

    def __init__(self, length: int = 10, source: Handler[Any, float | None] | None = None):
        """Initialize TRIMA handler with specified parameters.

        :param length: The period for the TRIMA calculation, defaults to 10
        :param source: Input data source, defaults to None
        """
        super().__init__(source=source)
        self.length = length if length and length > 0 else 10
        self.half_length = round(0.5 * (self.length + 1))

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields TRIMA values.

        This method implements the TRIMA calculation by creating a pipeline of two
        consecutive SMA calculations, each with half_length. This approach is
        mathematically equivalent to a weighted moving average with triangular weights.

        :return: Iterator yielding TRIMA values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        yield from self.source | SMAHandler(length=self.half_length) | SMAHandler(length=self.half_length)
