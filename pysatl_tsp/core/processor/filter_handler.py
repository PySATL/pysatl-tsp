from collections import deque
from collections.abc import Iterator
from typing import Any, Callable

from pysatl_tsp.core import Handler, T, U
from pysatl_tsp.core.scrubber import ScrubberWindow


class OnlineFilterHandler(Handler[T, U]):
    """A handler that applies a filter function to time series data in real-time.

    This handler processes data points one by one as they arrive and applies a filter
    function to the accumulated history. It's suitable for implementing online filters
    such as moving averages, exponential smoothing, or real-time anomaly detection.

    The filter function receives the current history window and configuration parameters,
    and produces a filtered value for each input value.

    :param filter_func: Function that applies filtering on the history window
    :param filter_config: Configuration parameters for the filter function, defaults to None
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        from pysatl_tsp.core.data_providers import SimpleDataProvider
        from pysatl_tsp.core.scrubber import ScrubberWindow
        from pysatl_tsp.core.processor import OnlineFilterHandler
        import random

        random.seed(42)
        data = [10 + i + random.uniform(-2, 2) for i in range(20)]
        data_source = SimpleDataProvider(data)


        # Define a simple moving average filter
        def moving_avg(window: ScrubberWindow[float], config: int) -> float:
            # Use only the last 'config' elements or all if less available
            lookback = min(len(window), config)
            if lookback == 0:
                return 0
            return sum(window[-lookback:].values) / lookback


        # Create the online filter with a window size of 5
        filter_handler = OnlineFilterHandler(filter_func=moving_avg, filter_config=5, source=data_source)

        # Process the data
        original_values = []
        filtered_values = []

        for i, filtered_value in enumerate(filter_handler):
            original_values.append(data[i])
            filtered_values.append(filtered_value)

        print("Original vs Filtered:")
        for orig, filt in zip(original_values[:10], filtered_values[:10]):
            print(f"{orig:.2f} -> {filt:.2f}")

        # Output might look like:
        # Original vs Filtered:
        # 9.67 -> 9.67
        # 11.79 -> 10.73
        # 11.56 -> 11.01
        # 12.89 -> 11.48
        # 13.89 -> 11.96
        # ...
        ```
    """

    def __init__(
        self,
        filter_func: Callable[[ScrubberWindow[T], Any], U],
        filter_config: Any = None,
        source: Handler[Any, T] | None = None,
    ):
        """Initialize an online filter handler.

        :param filter_func: Function that applies filtering on the history window
        :param filter_config: Configuration parameters for the filter function, defaults to None
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.filter_func = filter_func
        self.filter_config = filter_config

    def __iter__(self) -> Iterator[U]:
        """Create an iterator that yields filtered values in real-time.

        This method processes data points one by one, accumulates them in a history window,
        and applies the filter function to produce filtered values.

        :return: Iterator yielding filtered values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        self._history: ScrubberWindow[T] = ScrubberWindow()

        for item in self.source:
            self._history.append(item)
            yield self.filter_func(self._history, self.filter_config)


class OfflineFilterHandler(Handler[T, U]):
    """A handler that applies a filter function to the entire time series data in batch mode.

    This handler collects all data from the source before applying the filter function
    to the complete series at once. It's suitable for implementing filters that require
    the entire context of the time series, such as spectral filters, Savitzky-Golay filters,
    or other techniques that need to process the data as a whole.

    :param filter_func: Function that processes the entire series and returns filtered values
    :param filter_config: Configuration parameters for the filter function, defaults to None
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        # Create a data source
        import numpy as np
        from scipy import signal

        # Generate a noisy signal
        t = np.linspace(0, 1, 100)
        clean_signal = np.sin(2 * np.pi * 5 * t)
        noise = np.random.normal(0, 0.2, 100)
        noisy_signal = clean_signal + noise

        data_source = SimpleDataProvider(noisy_signal)


        # Define a Savitzky-Golay filter function
        def savgol_filter(window: ScrubberWindow[float], config: dict) -> list[float]:
            data = np.array(window.values)
            window_length = config.get("window_length", 11)
            polyorder = config.get("polyorder", 3)

            filtered = signal.savgol_filter(data, window_length, polyorder)
            return filtered.tolist()


        # Create the offline filter
        filter_handler = OfflineFilterHandler(
            filter_func=savgol_filter, filter_config={"window_length": 11, "polyorder": 3}, source=data_source
        )

        # Process and visualize the results
        filtered_signal = list(filter_handler)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(t, noisy_signal, "b", label="Noisy signal")
        plt.plot(t, filtered_signal, "r", label="Filtered signal")
        plt.plot(t, clean_signal, "g", label="Original clean signal")
        plt.legend()
        plt.show()
        ```
    """

    def __init__(
        self,
        filter_func: Callable[[ScrubberWindow[T], Any], list[U]],
        filter_config: Any = None,
        source: Handler[Any, T] | None = None,
    ):
        """Initialize an offline filter handler.

        :param filter_func: Function that processes the entire series and returns filtered values
        :param filter_config: Configuration parameters for the filter function, defaults to None
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.filter_func = filter_func
        self.filter_config = filter_config

    def __iter__(self) -> Iterator[U]:
        """Create an iterator that yields filtered values after processing the entire series.

        This method collects all data from the source, applies the filter function
        to the complete series, and then yields the resulting filtered values.

        :return: Iterator yielding filtered values
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        full_series = ScrubberWindow(deque(self.source))
        filtered_series = self.filter_func(full_series, self.filter_config)

        yield from filtered_series
