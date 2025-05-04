from collections.abc import Iterable, Iterator

from .abstract import DataProvider, T


class SimpleDataProvider(DataProvider[T]):
    """A data provider that serves data from an in-memory iterable collection.

    This class implements a simple data provider that wraps around any iterable
    collection and makes it available as a data source in a processing pipeline.
    It's useful for testing, working with pre-loaded data, or creating pipelines
    that process data already in memory.

    :param data: An iterable collection containing the time series data
    """

    def __init__(self, data: Iterable[T]) -> None:
        """Initialize a simple data provider with an iterable data source.

        :param data: An iterable collection containing the time series data
        """
        super().__init__()
        self.data = data

    def __iter__(self) -> Iterator[T]:
        """Create an iterator over the provided data collection.

        This method simply yields items from the data collection that was passed
        during initialization, making them available to subsequent handlers in
        the processing pipeline.

        :return: An iterator yielding items from the data collection
        """
        yield from self.data
