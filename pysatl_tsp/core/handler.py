from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import (
    Any,
    Generic,
    TypeVar,
)

__all__ = ["Handler", "T", "U", "V"]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Handler(ABC, Generic[T, U]):
    """Abstract base class for time series processing handlers.

    This class implements a Chain of Responsibility pattern for processing time series data.
    Each Handler can be connected to a source handler and process its output data.
    Handlers can be combined using the pipe operator (|) to create processing pipelines.

    :param source: The handler to use as a data source, defaults to None
    """

    def __init__(self, source: Handler[Any, T] | None = None):
        """Initialize a handler with an optional source.

        :param source: The handler to use as a data source, defaults to None
        """
        self._source = source

    @property
    def source(self) -> Handler[Any, T] | None:
        """Get the source handler that provides input data to this handler.

        :return: The source handler or None if this is a root handler
        """
        return self._source

    @source.setter
    def source(self, value: Handler[Any, T]) -> None:
        """Set the source handler for this handler.

        :param value: The handler to use as a data source
        :raises RuntimeError: If the source has already been set
        """
        if self._source is not None:
            raise RuntimeError("Cannot change already setted source")
        self._source = value

    @abstractmethod
    def __iter__(self) -> Iterator[U]:
        """Create an iterator over the output data produced by this handler.

        Each subclass must implement this method to define how data is processed.

        :return: An iterator yielding processed data items
        """
        pass

    def __or__(self, other: Handler[U, V]) -> Pipeline[T, V]:
        """Combine this handler with another handler using the pipe operator.

        This allows for the creation of processing pipelines using syntax like:
        handler1 | handler2 | handler3

        :param other: The next handler in the pipeline
        :return: A Pipeline object connecting this handler to the other handler
        """
        return Pipeline(self, other)


class Pipeline(Handler[T, V]):
    """A composite handler that connects two handlers in sequence.

    The Pipeline takes output from the first handler and feeds it as input to the second handler.
    This class enables the creation of data processing chains, where each handler in the chain
    performs a specific transformation on the data.

    :param first: The first handler in the pipeline
    :param second: The second handler in the pipeline
    :raises ValueError: If the second handler already has a source configured
    """

    def __init__(self, first: Handler[T, U], second: Handler[U, V]):
        """Initialize a pipeline with two handlers.

        :param first: The first handler in the pipeline
        :param second: The second handler in the pipeline
        :raises ValueError: If the second handler already has a source configured
        """
        super().__init__()
        self.first = first

        if second.source is not None:
            raise ValueError(
                f"Cannot create Pipeline: second handler {type(second).__name__} "
                f"already has a source {type(second.source).__name__}. "
                "Use explicit set_source() or rebuild pipeline chain."
            )

        self.second = second
        self.second.source = self.first

    def __iter__(self) -> Iterator[V]:
        """Create an iterator that processes data through both handlers in sequence.

        :return: An iterator yielding data processed through both handlers
        """
        self.second_iterator = iter(self.second)
        return self.second_iterator
