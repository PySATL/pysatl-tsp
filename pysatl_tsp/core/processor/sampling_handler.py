from collections.abc import Iterator
from typing import Any, Callable, Optional

from pysatl_tsp.core import Handler, T
from pysatl_tsp.core.scrubber import OfflineSegmentationScrubber, OnlineSegmentationScrubber, ScrubberWindow

from .mapping_handler import MappingHandler


class OnlineSamplingHandler(Handler[T, T]):
    """A handler that samples time series data in real-time based on a condition.

    This handler uses segmentation to identify points where sampling should occur
    and extracts the last item from each segment. It processes data in real-time
    and is suitable for adaptive sampling strategies, where sampling decisions
    are made based on the recent history of the time series.

    :param sampling_rule: Function that decides when to take a sample
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source with steadily increasing values
        data = list(range(100))
        data_source = SimpleDataProvider(data)


        # Define a sampling rule that samples when the value changes by more than 5
        def significant_change(window: ScrubberWindow[int]) -> bool:
            if len(window) < 2:
                return False

            # Get last sample taken (first item in window) and current value
            last_sampled = window[0]
            current = window[-1]

            # Sample if change is significant
            return abs(current - last_sampled) >= 5


        # Create a sampling handler
        sampler = OnlineSamplingHandler(sampling_rule=significant_change, source=data_source)

        # Process and collect sampled points
        sampled_points = list(sampler)

        print(f"Original data points: {len(data)}")
        print(f"Sampled data points: {len(sampled_points)}")
        print(f"Sampled values: {sampled_points[:10]}...")

        # Output might look like:
        # Original data points: 100
        # Sampled data points: 20
        # Sampled values: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]...
        ```
    """

    def __init__(self, sampling_rule: Callable[[ScrubberWindow[T]], bool], source: Optional[Handler[Any, T]] = None):
        """Initialize an online sampling handler.

        :param sampling_rule: Function that decides when to take a sample
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.sampling_rule = sampling_rule

    def __iter__(self) -> Iterator[T]:
        """Create an iterator that yields sampled values based on the sampling rule.

        This method uses OnlineSegmentationScrubber to segment the data and a
        MappingHandler to extract the last item from each segment.

        :return: Iterator yielding sampled values
        :raises ValueError: If no source has been set (propagated from segmentation scrubber)
        """
        mapping_handler: MappingHandler[ScrubberWindow[T], T] = MappingHandler(map_func=lambda window: window[-1])
        pipeline = (
            OnlineSegmentationScrubber(segmentation_rule=self.sampling_rule, source=self.source) | mapping_handler
        )

        yield from pipeline


class OfflineSamplingHandler(Handler[T, T]):
    """A handler that samples time series data in batch mode based on identified indices.

    This handler processes the entire dataset to identify sampling points before
    extracting the samples. It's suitable for global sampling strategies that consider
    the entire time series context, such as selecting representative points or
    key points that preserve the overall shape of the data.

    :param sampling_rule: Function that analyzes the entire series and returns indices of points to sample
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        from typing import List

        # Create a data source with a sinusoidal signal
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x)
        data_source = SimpleDataProvider(y)

        # Define an offline sampling rule that selects local extrema
        def find_extrema(window: ScrubberWindow[float]) -> List[int]:
            data = np.array(window.values)
            # Find local maxima and minima
            extrema_indices = []

            # First point is always included
            extrema_indices.append(0)

            # Find local maxima and minima (simplified)
            for i in range(1, len(data)-1):
                if (data[i] > data[i-1] and data[i] > data[i+1]) or \
                   (data[i] < data[i-1] and data[i] < data[i+1]):
                    extrema_indices.append(i)

            # Last point is always included
            extrema_indices.append(len(data)-1)

            return extrema_indices

        # Create a sampling handler
        sampler = OfflineSamplingHandler(
            sampling_rule=find_extrema,
            source=data_source
        )

        # Process and collect sampled points
        sampled_indices = []
        sampled_values = []
        original_values = list(y)

        for i, value in enumerate(sampler):
            sampled_values.append(value)
            # Approximate index (not exact)
            sampled_indices.append(i * len(original_values) // len(sampled_values))

        # Visualize the results
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'b-', label='Original signal')
        plt.plot(x[sampled_indices], sampled_values, 'ro', label='Sampled points')
        plt.legend()
        plt.title('Sinusoidal Signal with Extrema Sampling')
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.grid(True)
        plt.show()

        print(f"Original data points: {len(original_values)}")
        print(f"Sampled data points: {len(sampled_values)}")
        ```
    """

    def __init__(
        self, sampling_rule: Callable[[ScrubberWindow[T]], list[int]], source: Optional[Handler[Any, T]] = None
    ):
        """Initialize an offline sampling handler.

        :param sampling_rule: Function that analyzes the entire series and returns indices of points to sample
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.sampling_rule = sampling_rule

    def __iter__(self) -> Iterator[T]:
        """Create an iterator that yields sampled values based on the indices identified by the sampling rule.

        This method uses OfflineSegmentationScrubber to segment the data at the specified indices
        and a MappingHandler to extract the last item from each segment.

        :return: Iterator yielding sampled values
        :raises ValueError: If no source has been set (propagated from segmentation scrubber)
        """
        mapping_handler: MappingHandler[ScrubberWindow[T], T] = MappingHandler(map_func=lambda window: window[-1])
        pipeline = (
            OfflineSegmentationScrubber(segmentation_rule=self.sampling_rule, source=self.source) | mapping_handler
        )

        yield from pipeline
