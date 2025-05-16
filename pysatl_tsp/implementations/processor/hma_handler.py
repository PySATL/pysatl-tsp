from __future__ import annotations

import math
from collections.abc import Iterator

from pysatl_tsp.core import Handler
from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.core.processor.combine_handler import CombineHandler
from pysatl_tsp.implementations.processor.wma_handler import WMAHandler


class HMAHandler(Handler[float | None, float | None]):
    """Hull Moving Average (HMA) handler.

    The Hull Moving Average is designed to reduce lag while maintaining smoothness.
    It uses weighted moving averages (WMA) in a multi-step process to create a more
    responsive indicator that better follows price action.

    The HMA is calculated using the following formula:
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    :param length: The period for HMA calculation

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Create a Hull Moving Average handler with length of 4
        hma_handler = HMAHandler(length=4)
        hma_handler.set_source(data_source)

        # Process the data
        for value in hma_handler:
            print(value)

        # The first few values may be None as the HMA needs historical data
        # Then the HMA values will follow, being more responsive than traditional
        # moving averages while maintaining smoothness.
        ```
    """

    def __init__(self, length: int):
        """Initialize a Hull Moving Average handler.

        :param length: The period for HMA calculation
        """
        super().__init__()
        self.length = length

    def __iter__(self) -> Iterator[float | None]:
        """Create an iterator that yields HMA values.

        This method constructs a pipeline that:
        1. Takes values from the source
        2. Calculates two WMAs with different periods (length//2 and length)
        3. Combines them using the formula: 2*WMA(length//2) - WMA(length)
        4. Applies another WMA with period=sqrt(length) to the result

        :return: Iterator yielding HMA values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        def combine_func(lst: list[float | None]) -> float | None:
            if lst[0] is None or lst[1] is None:
                return None
            return 2 * lst[0] - lst[1]

        yield from (
            SimpleDataProvider(self.source)
            | CombineHandler(combine_func, WMAHandler(length=self.length // 2), WMAHandler(length=self.length))
            | WMAHandler(length=int(math.sqrt(self.length)))
        )
