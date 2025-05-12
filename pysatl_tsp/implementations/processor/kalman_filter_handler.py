from typing import Any, Union

import numpy as np

from pysatl_tsp.core import Handler
from pysatl_tsp.core.processor import OnlineFilterHandler
from pysatl_tsp.core.scrubber import ScrubberWindow


class KalmanFilterHandler(OnlineFilterHandler[float, float]):
    """A handler that applies the Kalman filter to time series data in real-time.

    This handler integrates the complete Kalman filter functionality for processing
    noisy time series data. It estimates the underlying state of a system based on
    a sequence of noisy measurements.

    :param F: State transition matrix
    :param H: Measurement matrix
    :param B: Control input matrix, defaults to 0
    :param Q: Process noise covariance matrix, defaults to identity matrix
    :param R: Measurement noise covariance matrix, defaults to identity matrix
    :param P: Initial state covariance matrix, defaults to identity matrix
    :param x0: Initial state vector, defaults to zero vector
    :param source: The handler providing input data, defaults to None

    Example:
    ```
        import numpy as np
        from pysatl_tsp.core.data_providers import SimpleDataProvider
        from pysatl_tsp.implementations.processor.kalman_filter_handler import KalmanFilterHandler

        np.random.seed(42)
        true_signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
        noisy_signal = true_signal + np.random.normal(0, 0.1, 1000)

        data_source = SimpleDataProvider(noisy_signal.tolist())

        dt: float = 1.0/60
        F = np.array([[1, dt, 0],[0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)

        filter_handler: KalmanFilterHandler = KalmanFilterHandler(F=F, H=H, Q=Q, R=R, source=data_source)

        filtered_values = list(filter_handler)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(noisy_signal)), noisy_signal, label='Measurements')
        plt.plot(range(len(filtered_values)), filtered_values, label='Kalman Filter Prediction')
        plt.legend()
        plt.show()
    ```
    """

    def __init__(
        self,
        F: np.ndarray[Any, np.dtype[np.float64]],
        H: np.ndarray[Any, np.dtype[np.float64]],
        B: Union[float, np.ndarray[Any, np.dtype[np.float64]]] | None = None,
        Q: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        R: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        P: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        x0: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        source: Handler[Any, float] | None = None,
    ) -> None:
        """Initialize the Kalman filter handler with all necessary matrices.

        :param F: State transition matrix
        :param H: Measurement matrix
        :param B: Control input matrix, defaults to None
        :param Q: Process noise covariance matrix, defaults to None
        :param R: Measurement noise covariance matrix, defaults to None
        :param P: Initial state covariance matrix, defaults to None
        :param x0: Initial state vector, defaults to None
        :param source: The handler providing input data, defaults to None
        """
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n: int = F.shape[1]
        self.m: int = H.shape[1]

        self.F: np.ndarray[Any, np.dtype[np.float64]] = F
        self.H: np.ndarray[Any, np.dtype[np.float64]] = H
        self.B: Union[float, np.ndarray[Any, np.dtype[np.float64]]] = 0 if B is None else B

        self.Q: np.ndarray[Any, np.dtype[np.float64]] = np.eye(self.n) if Q is None else Q
        self.R: np.ndarray[Any, np.dtype[np.float64]] = np.eye(self.n) if R is None else R
        self.P: np.ndarray[Any, np.dtype[np.float64]] = np.eye(self.n) if P is None else P

        self.x: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((self.n, 1)) if x0 is None else x0

        super().__init__(filter_func=self._apply_kalman_filter, filter_config=None, source=source)

    def predict(
        self, u: Union[float, np.ndarray[Any, np.dtype[np.float64]]] = 0
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Predict the next state based on the model.

        :param u: Control input, defaults to 0
        :return: Predicted state vector
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)

        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x

    def update(self, z: float) -> None:
        """Update the state estimate based on the measurement.

        :param z: Measurement
        """
        y: np.ndarray[Any, np.dtype[np.float64]] = z - np.dot(self.H, self.x)

        S: np.ndarray[Any, np.dtype[np.float64]] = self.R + np.dot(self.H, np.dot(self.P, self.H.T))

        K: np.ndarray[Any, np.dtype[np.float64]] = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)

        I_matrix: np.ndarray[Any, np.dtype[np.float64]] = np.eye(self.n)
        self.P = np.dot(np.dot(I_matrix - np.dot(K, self.H), self.P), (I_matrix - np.dot(K, self.H)).T) + np.dot(
            np.dot(K, self.R), K.T
        )

    def _apply_kalman_filter(self, window: ScrubberWindow[float], _: Any) -> float:
        """Apply the Kalman filter to the latest point in the window.

        :param window: Window of historical data
        :param _: Unused configuration parameter
        :return: Filtered value
        """
        if not window:
            return 0.0

        measurement: float = window[-1]

        prediction_array = np.dot(self.H, self.predict())
        prediction: float = float(prediction_array.item())

        self.update(measurement)

        return prediction
