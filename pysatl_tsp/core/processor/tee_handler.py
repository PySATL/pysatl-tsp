import itertools
from collections.abc import Iterator
from typing import Callable, TypeVar

from pysatl_tsp.core import Handler, T, U
from pysatl_tsp.core.data_providers import SimpleDataProvider

S = TypeVar("S")


class TeeHandler(Handler[T, U]):
    """A handler that processes data through two parallel paths and combines the results.

    This handler takes input data from a source, sends it through both the original path
    and a processing path simultaneously, and then combines each pair of outputs using
    a provided function. It's useful for operations where you need to preserve the original
    data while also using a transformed version of it.

    :param processor: Handler that processes the tee'd data stream
    :param combine_func: Function that combines original and processed values

    Example:
        ```python
        # Create a data source
        data_source = SimpleDataProvider([1, 2, 3, 4, 5])

        # Define a processor that squares the values
        square_processor = MappingHandler(map_func=lambda x: x * x)


        # Define a function to combine original and processed values
        def combine(original, processed):
            return f"{original} squared is {processed}"


        # Create and use the tee handler
        tee_handler = TeeHandler(processor=square_processor, combine_func=combine)
        tee_handler.set_source(data_source)

        # Process the data
        for result in tee_handler:
            print(result)

        # Output:
        # 1 squared is 1
        # 2 squared is 4
        # 3 squared is 9
        # 4 squared is 16
        # 5 squared is 25
        ```
    """

    def __init__(self, processor: Handler[T, S], combine_func: Callable[[T, S], U]):
        """Initialize a tee handler.

        :param processor: Handler that processes the tee'd data stream
        :param combine_func: Function that combines original and processed values
        """
        super().__init__()
        self.processor = processor
        self.combine_func = combine_func

    def __iter__(self) -> Iterator[U]:
        """Create an iterator that yields combined results from original and processed data.

        This method creates two identical iterators from the source, processes one through
        the processor, and then combines corresponding items from both streams using the
        combine function.

        :return: Iterator yielding combined results
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("TeeHandler requires a data source")

        source_iterator = iter(self.source)

        original_iter, process_iter = itertools.tee(source_iterator)

        process_provider = SimpleDataProvider(process_iter)
        processed_pipeline = process_provider | self.processor
        processed_iter = iter(processed_pipeline)

        try:
            while True:
                original_value = next(original_iter)
                processed_value = next(processed_iter)
                yield self.combine_func(original_value, processed_value)
        except StopIteration:
            pass
