from collections.abc import Iterator
from typing import Callable, TypeVar

from .abstract import DataProvider

X = TypeVar("X")


class FileDataProvider(DataProvider[X]):
    """A data provider that reads time series data from a text file.

    This class implements a file-based data source that reads a file line by line
    and transforms each line into a data item using a specified handler function.
    It is useful for processing time series data stored in text files with various
    formats (CSV, JSON per line, custom formats, etc.).

    :param filename: Path to the file containing time series data
    :param handler: A function that converts each line of text to a data item

    :raises FileNotFoundError: If the specified file does not exist
    :raises PermissionError: If the file cannot be read due to permission issues
    """

    def __init__(self, filename: str, handler: Callable[[str], X]) -> None:
        """Initialize a file data provider.

        :param filename: Path to the file containing time series data
        :param handler: A function that converts each line of text to a data item
        """
        super().__init__()
        self.filename = filename
        self.handler = handler

    def __iter__(self) -> Iterator[X]:
        """Create an iterator that reads the file line by line and yields processed data items.

        The method opens the specified file, reads it line by line, and applies the handler
        function to each line to transform it into a data item.

        :return: An iterator yielding processed data items from the file
        :raises FileNotFoundError: If the specified file does not exist
        :raises PermissionError: If the file cannot be read due to permission issues
        """
        with open(self.filename) as f:
            for line in f:
                yield self.handler(line)
