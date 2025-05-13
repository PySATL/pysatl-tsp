import itertools
from collections.abc import Iterator
from typing import Any, Callable

from pysatl_tsp.core import Handler, T, U
from pysatl_tsp.core.data_providers import SimpleDataProvider


class CombineHandler(Handler[T, U]):
    """A handler that combines outputs from multiple handlers processing the same data.

    This handler feeds the same source data to multiple handlers in parallel and combines
    their outputs using a user-provided function. It's useful for scenarios where you need
    to process the same data in different ways and then merge the results, such as feature
    extraction, multi-model predictions, or parallel transformations.

    :param combine_func: Function that combines values from all handlers into a single output
    :param handlers: Variable number of handlers whose outputs will be combined
    :param continue_on_partial: Whether to continue when some handlers are exhausted, defaults to True

    Example:
        ```python
        # Create a data source
        data_source = SimpleDataProvider([1, 2, 3, 4, 5])

        # Define handlers for different transformations
        square_handler = MappingHandler(map_func=lambda x: x * x)
        double_handler = MappingHandler(map_func=lambda x: 2 * x)
        str_handler = MappingHandler(map_func=lambda x: f"Value: {x}")

        # Function to combine outputs from all handlers
        def combine_results(values):
            return {
                "original^2": values[0],
                "original*2": values[1],
                "string": values[2]
            }

        # Create and use the combine handler
        combine = CombineHandler(
            combine_func=combine_results,
            square_handler, double_handler, str_handler
        )
        combine.set_source(data_source)

        # Process the data
        for result in combine:
            print(result)

        # Output:
        # {'original^2': 1, 'original*2': 2, 'string': 'Value: 1'}
        # {'original^2': 4, 'original*2': 4, 'string': 'Value: 2'}
        # {'original^2': 9, 'original*2': 6, 'string': 'Value: 3'}
        # {'original^2': 16, 'original*2': 8, 'string': 'Value: 4'}
        # {'original^2': 25, 'original*2': 10, 'string': 'Value: 5'}
        ```
    """

    def __init__(
        self, combine_func: Callable[[list[Any]], U], *handlers: Handler[T, Any], continue_on_partial: bool = True
    ):
        """Initialize a combine handler.

        :param combine_func: Function that combines values from all handlers into a single output
        :param handlers: Variable number of handlers whose outputs will be combined
        :param continue_on_partial: Whether to continue when some handlers are exhausted, defaults to True
        """
        super().__init__()
        self.handlers = handlers
        self.combine_func = combine_func
        self.continue_on_partial = continue_on_partial

    def __iter__(self) -> Iterator[U]:
        """Create an iterator that yields combined results from multiple handlers.

        This method processes the source data through each handler in parallel and
        combines their outputs using the combine function. If continue_on_partial is True,
        it will continue producing outputs even after some handlers are exhausted (using None
        for exhausted handlers). If False, it will stop when any handler is exhausted.

        :return: Iterator yielding combined results
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        source_iterators = itertools.tee(iter(self.source), len(self.handlers))

        pipelines = []
        for i, handler in enumerate(self.handlers):
            data_provider = SimpleDataProvider(source_iterators[i])
            pipelines.append(data_provider | handler)

        iterators = [iter(pipeline) for pipeline in pipelines]

        active_iterators = [True] * len(iterators)

        while any(active_iterators):
            values = [None] * len(iterators)

            for i, iterator in enumerate(iterators):
                if active_iterators[i]:
                    try:
                        values[i] = next(iterator)
                    except StopIteration:
                        active_iterators[i] = False

            if not self.continue_on_partial and not all(active_iterators):
                break

            if not any(active_iterators):
                break

            yield self.combine_func(values)
