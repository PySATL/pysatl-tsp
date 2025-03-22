from collections.abc import Iterator
from typing import Any, Callable, Optional

from ts_flow.core import Handler, T, U


class MappingHandler(Handler[T, U]):
    def __init__(self, map_func: Callable[[T], U], source: Optional[Handler[Any, T]] = None):
        super().__init__(source)
        self.map_func = map_func

    def __iter__(self) -> Iterator[U]:
        if self.source is None:
            raise ValueError("Source is not set")

        for segment in self.source:
            yield self.map_func(segment)
