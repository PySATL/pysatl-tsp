from abc import abstractmethod
from collections.abc import Iterator

from pysatl_tsp.core import Handler, T

__all__ = ["DataProvider", "T"]


class DataProvider(Handler[None, T]):
    """Abstract base class for time series data providers.

    DataProvider serves as a root handler in a processing pipeline, responsible for
    sourcing the initial time series data. As the first element in the chain, it doesn't
    receive input from any preceding handler and acts as the data origin.

    This class is designed to be subclassed with specific implementations for different
    data sources such as files, databases, APIs, or generated data.
    """

    def __init__(self) -> None:
        """Initialize a data provider with no source.

        Data providers are always root handlers and cannot have a source.
        """
        super().__init__(source=None)

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Create an iterator over the time series data provided by this data source.

        Each subclass must implement this method to define how data is sourced and
        potentially pre-processed before being passed to subsequent handlers.

        :return: An iterator yielding time series data items
        """
        pass
