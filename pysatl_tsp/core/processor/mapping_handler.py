from collections.abc import Iterator
from typing import Any, Callable

from pysatl_tsp.core import Handler, T, U


class MappingHandler(Handler[T, U]):
    """A handler that transforms time series data by applying a mapping function to each item.

    This handler applies a user-defined transformation function to each data point
    in the input stream, producing a new stream of transformed values. It's useful for
    simple point-by-point transformations such as scaling, type conversion, feature
    extraction, or any operation that processes one input item at a time.

    :param map_func: Function that transforms each input item to an output item
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source with numeric values
        data_source = SimpleDataProvider([1, 2, 3, 4, 5])


        # Simple mapping function to square each value
        def square(x: int) -> int:
            return x * x


        # Create a mapping handler
        mapper = MappingHandler(map_func=square, source=data_source)

        # Process the data
        for transformed in mapper:
            print(transformed)

        # Output:
        # 1
        # 4
        # 9
        # 16
        # 25

        # Example with a more complex transformation
        import json

        # Data source with JSON strings
        json_data = [
            '{"timestamp": "2023-09-01T10:00:00", "value": 42.5}',
            '{"timestamp": "2023-09-01T10:01:00", "value": 43.2}',
            '{"timestamp": "2023-09-01T10:02:00", "value": 41.8}',
        ]
        json_source = SimpleDataProvider(json_data)


        # Function to extract timestamp and value from JSON
        def parse_json(json_str: str) -> tuple[str, float]:
            data = json.loads(json_str)
            return (data["timestamp"], data["value"])


        # Create a mapping handler for JSON parsing
        json_mapper = MappingHandler(map_func=parse_json, source=json_source)

        # Process JSON data
        for timestamp, value in json_mapper:
            print(f"Time: {timestamp}, Value: {value}")

        # Output:
        # Time: 2023-09-01T10:00:00, Value: 42.5
        # Time: 2023-09-01T10:01:00, Value: 43.2
        # Time: 2023-09-01T10:02:00, Value: 41.8
        ```
    """

    def __init__(self, map_func: Callable[[T], U], source: Handler[Any, T] | None = None):
        """Initialize a mapping handler.

        :param map_func: Function that transforms each input item to an output item
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.map_func = map_func

    def __iter__(self) -> Iterator[U]:
        """Create an iterator that yields transformed items.

        This method iterates through the source data and applies the mapping function
        to each item, yielding the transformed results.

        :return: Iterator yielding transformed items
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        for segment in self.source:
            yield self.map_func(segment)
