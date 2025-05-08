from collections.abc import Iterator
from typing import Any, Optional

from pysatl_tsp.core import Handler, T
from pysatl_tsp.core.processor import MappingHandler
from pysatl_tsp.core.scrubber import ScrubberWindow, SlidingScrubber


class TimeSeriesCrossValidator(Handler[T, tuple[ScrubberWindow[T], ScrubberWindow[T]]]):
    """A handler that implements expanding window cross-validation for time series data.

    This handler produces a sequence of train-validation splits suitable for time series
    validation, where each split preserves the temporal order of data. It implements an
    expanding window approach, where the training set grows over time while the validation
    set has a fixed size and slides forward.

    The handler ensures that:
    1. The training set always has at least `min_train_size` points
    2. The validation set always has exactly `val_size` points
    3. The validation set always follows the training set temporally
    4. Each new split adds `val_size` points to the training set

    This approach respects the temporal nature of time series data and prevents
    data leakage from future to past.

    :param min_train_size: Minimum number of points in the initial training set
    :param val_size: Number of points in each validation set
    :param source: The handler providing input data, defaults to None

    Example:
        ```python
        import numpy as np
        import matplotlib.pyplot as plt

        # Generate a synthetic time series
        np.random.seed(42)
        ts = np.cumsum(np.random.normal(0, 1, 100))  # Random walk
        data_source = SimpleDataProvider(ts)

        # Create a cross-validator with min_train_size=50 and val_size=10
        cv = TimeSeriesCrossValidator(min_train_size=50, val_size=10, source=data_source)

        # Visualize the different train-validation splits
        plt.figure(figsize=(14, 8))
        x = np.arange(len(ts))
        plt.plot(x, ts, "k-", alpha=0.3, label="Full time series")

        for i, (train, val) in enumerate(cv):
            train_indices = list(train.indices)
            val_indices = list(val.indices)

            # Plot each split
            plt.plot(train_indices, [ts[i] for i in train_indices], "b-", linewidth=2, alpha=0.7 - i * 0.1)
            plt.plot(val_indices, [ts[i] for i in val_indices], "r-", linewidth=2, alpha=0.7 - i * 0.1)

            # Add markers at the split point
            split_idx = train_indices[-1]
            plt.axvline(x=split_idx, color="g", linestyle="--", alpha=0.5)

            # Print information about this split
            print(f"Split {i + 1}:")
            print(f"  Train: {len(train)} points (indices {train_indices[0]}..{train_indices[-1]})")
            print(f"  Validation: {len(val)} points (indices {val_indices[0]}..{val_indices[-1]})")

        plt.title("Time Series Cross-Validation: Expanding Window Approach")
        plt.xlabel("Time")
        plt.ylabel("Value")

        # Add custom legend
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="k", alpha=0.3),
            Line2D([0], [0], color="b", linewidth=2),
            Line2D([0], [0], color="r", linewidth=2),
            Line2D([0], [0], color="g", linestyle="--"),
        ]
        plt.legend(
            custom_lines, ["Full time series", "Training sets", "Validation sets", "Split points"], loc="upper left"
        )

        plt.grid(True, alpha=0.3)
        plt.show()

        # Example model evaluation with each split
        from sklearn.linear_model import LinearRegression

        for i, (train, val) in enumerate(cv):
            # Prepare data
            train_indices = list(train.indices)
            train_X = np.array(train_indices).reshape(-1, 1)
            train_y = np.array(list(train.values))

            val_indices = list(val.indices)
            val_X = np.array(val_indices).reshape(-1, 1)
            val_y = np.array(list(val.values))

            # Train a simple model
            model = LinearRegression()
            model.fit(train_X, train_y)

            # Evaluate on validation set
            val_pred = model.predict(val_X)
            mse = np.mean((val_pred - val_y) ** 2)

            print(f"Split {i + 1} - Validation MSE: {mse:.4f}")
        ```
    """

    def __init__(self, min_train_size: int, val_size: int, source: Optional[Handler[Any, T]] = None):
        """Initialize a time series cross-validator.

        :param min_train_size: Minimum number of points in the initial training set
        :param val_size: Number of points in each validation set
        :param source: The handler providing input data, defaults to None
        """
        super().__init__(source)
        self.min_train_size = min_train_size
        self.val_size = val_size

    def __iter__(self) -> Iterator[tuple[ScrubberWindow[T], ScrubberWindow[T]]]:
        """Create an iterator that yields train-validation splits for time series cross-validation.

        This method creates splits where:
        1. The first split has exactly min_train_size points for training
        2. Each subsequent split adds val_size points to the training set
        3. Each validation set has exactly val_size points and follows the training set

        :return: Iterator yielding tuples of (training_window, validation_window)
        :raises ValueError: If no source has been set
        """
        if self.source is None:
            raise ValueError("Source is not set")

        scrubber = SlidingScrubber(
            lambda buffer: len(buffer) > self.min_train_size
            and (len(buffer) - self.min_train_size) % self.val_size == 0,
            shift=0,
            source=self.source,
        )
        handler: MappingHandler[ScrubberWindow[T], tuple[ScrubberWindow[T], ScrubberWindow[T]]] = MappingHandler(
            map_func=lambda window: (window[: -self.val_size], window[-self.val_size :])
        )

        return iter(scrubber | handler)
