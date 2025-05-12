from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations import KalmanFilterHandler, TimeSeriesCrossValidator


class TestTimeSeriesCrossValidator:
    def test_basic_functionality(self) -> None:
        data = list(range(20))
        provider = SimpleDataProvider(data)

        min_train_size = 10
        val_size = 2
        cv = TimeSeriesCrossValidator(min_train_size=min_train_size, val_size=val_size, source=provider)

        splits = list(cv)

        assert len(splits) == (len(data) - min_train_size) // val_size

        # Check the first split
        train1, val1 = splits[0]
        assert len(train1) == min_train_size
        assert len(val1) == val_size
        assert list(train1.values) == list(range(10))
        assert list(val1.values) == [10, 11]

        # Check the last split
        train_last, val_last = splits[-1]
        assert len(train_last) == len(data) - val_size
        assert len(val_last) == val_size
        assert list(train_last.values) == list(range(18))
        assert list(val_last.values) == [18, 19]

    def test_raises_error_when_source_not_set(self) -> None:
        cv: TimeSeriesCrossValidator[Any] = TimeSeriesCrossValidator(min_train_size=10, val_size=2)

        with pytest.raises(ValueError, match="Source is not set"):
            next(iter(cv))

    @given(
        min_train_size=st.integers(min_value=1, max_value=20),
        val_size=st.integers(min_value=1, max_value=10),
        extra_data_points=st.integers(min_value=0, max_value=100),
    )
    def test_property_based(self, min_train_size: int, val_size: int, extra_data_points: int) -> None:
        total_data_size = min_train_size + val_size + extra_data_points
        data = list(range(total_data_size))

        provider = SimpleDataProvider(data)
        cv = TimeSeriesCrossValidator(min_train_size=min_train_size, val_size=val_size, source=provider)

        splits = list(cv)

        expected_splits = 1 + extra_data_points // val_size
        assert len(splits) == expected_splits

        for i, (train, val) in enumerate(splits):
            assert len(train) == min_train_size + i * val_size
            assert len(val) == val_size

            expected_val_start = min_train_size + i * val_size
            assert list(val.values) == data[expected_val_start : expected_val_start + val_size]
            assert list(train.values) == data[:expected_val_start]

    @given(data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=15, max_size=50))
    def test_window_indices_preserved(self, data: list[float]) -> None:
        provider = SimpleDataProvider(data)
        cv = TimeSeriesCrossValidator(min_train_size=10, val_size=5, source=provider)

        for train, val in cv:
            assert list(train.indices) == list(range(len(train)))
            assert list(val.indices) == list(range(len(train), len(train) + len(val)))
            assert train.indices[-1] + 1 == val.indices[0]

    def test_empty_data(self) -> None:
        provider: SimpleDataProvider[Any] = SimpleDataProvider([])
        cv = TimeSeriesCrossValidator(min_train_size=5, val_size=2, source=provider)

        assert list(cv) == []

    def test_insufficient_data(self) -> None:
        provider = SimpleDataProvider(list(range(9)))
        cv = TimeSeriesCrossValidator(min_train_size=10, val_size=2, source=provider)

        assert list(cv) == []

    def test_exact_data_for_one_split(self) -> None:
        provider = SimpleDataProvider(list(range(12)))
        min_train_size = 10
        val_size = 2
        cv = TimeSeriesCrossValidator(min_train_size=min_train_size, val_size=val_size, source=provider)

        splits = list(cv)
        assert len(splits) == 1

        train, val = splits[0]
        assert len(train) == min_train_size
        assert len(val) == val_size
        assert list(train.values) == list(range(10))
        assert list(val.values) == [10, 11]


class TestKalmanFilterHandler:
    def test_basic_functionality(self) -> None:
        # Create a simple linear signal with noise
        x = np.linspace(0, 10, 100)
        true_signal = 2 * x + 1
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 1, 100)
        measurements = true_signal + noise

        # Set up the Kalman filter for a constant velocity model
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])  # State transition matrix
        H = np.array([1, 0]).reshape(1, 2)  # Measurement matrix
        Q = np.array([[0.01, 0.01], [0.01, 0.04]])  # Process noise covariance
        R = np.array([1.0]).reshape(1, 1)  # Measurement noise covariance

        # Create the filter handler
        provider = SimpleDataProvider(measurements.tolist())
        filter_handler = KalmanFilterHandler(F=F, H=H, Q=Q, R=R, source=provider)

        # Apply the filter
        filtered_values = list(filter_handler)

        # Check that filtering improved the signal
        # (Mean squared error should be less for filtered values)
        mse_raw = np.mean((measurements - true_signal) ** 2)
        mse_filtered = np.mean((np.array(filtered_values) - true_signal) ** 2)

        assert mse_filtered < mse_raw
        assert len(filtered_values) == len(measurements)

    def test_empty_data(self) -> None:
        # Set up the Kalman filter
        F = np.array([[1, 1], [0, 1]])
        H = np.array([1, 0]).reshape(1, 2)

        # Create the filter handler with empty data
        lst: list[Any] = []
        provider = SimpleDataProvider(lst)
        filter_handler = KalmanFilterHandler(F=F, H=H, source=provider)

        # The filter should return an empty list
        assert list(filter_handler) == []

    def test_raises_error_when_source_not_set(self) -> None:
        # Create filter without a source
        F = np.array([[1, 1], [0, 1]])
        H = np.array([1, 0]).reshape(1, 2)

        filter_handler = KalmanFilterHandler(F=F, H=H)

        with pytest.raises(ValueError, match="Source is not set"):
            next(iter(filter_handler))

    def test_predict_method(self) -> None:
        """Test the predict method directly."""
        F = np.array([[1, 1], [0, 1]])
        H = np.array([1, 0]).reshape(1, 2)
        x0 = np.array([[5], [2]])  # Initial state (position=5, velocity=2)

        kf = KalmanFilterHandler(F=F, H=H, x0=x0)

        # Predict should advance state by F*x
        predicted = kf.predict()
        assert predicted.shape == (2, 1)
        assert np.isclose(predicted[0, 0], 7)  # position + velocity = 5 + 2
        assert np.isclose(predicted[1, 0], 2)  # velocity unchanged

    def test_update_method(self) -> None:
        """Test the update method directly."""
        F = np.array([[1]])
        H = np.array([1]).reshape(1, 1)
        P = np.array([[1]])
        R = np.array([0.5]).reshape(1, 1)
        x0 = np.array([[0]])  # Initial state = 0

        kf = KalmanFilterHandler(F=F, H=H, P=P, R=R, x0=x0)

        # First predict (should still be 0)
        kf.predict()

        # Update with measurement = 10
        measurement = 10.0
        kf.update(measurement)

        # State should move toward measurement
        assert kf.x[0, 0] > 0  # Should be greater than initial
        assert kf.x[0, 0] < measurement  # Should be less than measurement

    def test_dimensions(self) -> None:
        """Test with different state dimensions."""
        # 2D state
        F_2d = np.array([[1, 1], [0, 1]])
        H_2d = np.array([1, 0]).reshape(1, 2)

        kf_2d = KalmanFilterHandler(F=F_2d, H=H_2d)
        assert kf_2d.n == len(F_2d)
        assert kf_2d.x.shape == (2, 1)

        # 3D state
        F_3d = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])
        H_3d = np.array([1, 0, 0]).reshape(1, 3)

        kf_3d = KalmanFilterHandler(F=F_3d, H=H_3d)
        assert kf_3d.n == len(F_3d)
        assert kf_3d.x.shape == (3, 1)

    def test_known_case(self) -> None:
        """Test with a known simple case."""
        # Simple case: Constant signal with noise
        measurements = [5.1, 4.9, 5.2, 4.8, 5.1]  # Measurements around 5.0

        F = np.array([[1]])
        H = np.array([1]).reshape(1, 1)
        Q = np.array([[0.001]])
        R = np.array([0.1]).reshape(1, 1)
        P = np.array([[1]])
        x0 = np.array([[0]])  # Start with zero

        provider = SimpleDataProvider(measurements)
        filter_handler = KalmanFilterHandler(F=F, H=H, Q=Q, R=R, P=P, x0=x0, source=provider)

        filtered_values = list(filter_handler)

        assert len(filtered_values) == len(measurements)
        assert filtered_values[0] == 0

        errors = [abs(v - 5.0) for v in filtered_values]
        assert errors[-1] < errors[0]

    def test_initialization_with_matrices(self) -> None:
        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([1, 0]).reshape(1, 2)
        B = np.array([[0.05], [0.1]])
        Q = np.array([[0.01, 0], [0, 0.01]])
        R = np.array([0.1]).reshape(1, 1)
        P = np.array([[1, 0], [0, 1]])
        x0 = np.array([[1], [0]])

        kf = KalmanFilterHandler(F=F, H=H, B=B, Q=Q, R=R, P=P, x0=x0)

        assert np.array_equal(kf.F, F)
        assert np.array_equal(kf.H, H)
        assert np.array_equal(kf.B, B)
        assert np.array_equal(kf.Q, Q)
        assert np.array_equal(kf.R, R)
        assert np.array_equal(kf.P, P)
        assert np.array_equal(kf.x, x0)

    @given(
        data=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=5, max_size=50
        )
    )
    def test_filter_handles_arbitrary_data(self, data: list[float]) -> None:
        """Test that filter can handle arbitrary data sequences."""
        # Simple filter
        F = np.array([[1]])
        H = np.array([1]).reshape(1, 1)

        provider = SimpleDataProvider(data)
        filter_handler = KalmanFilterHandler(F=F, H=H, source=provider)

        filtered_values = list(filter_handler)
        assert len(filtered_values) == len(data)
