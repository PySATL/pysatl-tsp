import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.sma_handler import SMAHandler
from tests.utils import safe_allclose


@given(
    data=st.lists(
        st.one_of(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            st.none(),
        ),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=1, max_value=20),
    min_periods=st.integers(min_value=1, max_value=20) | st.none(),
)
def test_sma_calculation_property_based(data: list[float | None], length: int, min_periods: int | None) -> None:
    if length > len(data):
        return

    # If min_periods is None, it defaults to length in both implementations
    effective_min_periods = min(min_periods, length) if min_periods is not None else length

    # If all values are None, no meaningful test can be performed
    if all(x is None for x in data):
        data = [*data, 1.0]  # Add at least one non-None value

    # Calculate reference values using pandas_ta
    pta = ta.sma(pd.Series(data), length=length, min_periods=effective_min_periods)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | SMAHandler(length=length, min_periods=effective_min_periods))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data={data}, length={length}, min_periods={effective_min_periods}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data, length, min_periods",
    [
        # Basic scenarios
        ([1, 2, 3, 4, 5], 3, None),  # Simple case, min_periods defaults to length
        ([1, 2, 3, 4, 5], 3, 1),  # Lower min_periods than length
        ([1, 2, 3, 4, 5], 3, 3),  # min_periods equals length
        ([1, 2, 3, 4, 5], 5, None),  # Length equals data length
        ([1, 2, 3, 4, 5], 1, None),  # Length = 1
        # None handling scenarios
        ([None, 2, 3, 4, 5], 3, None),  # None at the beginning
        ([1, 2, 3, None, 5], 3, None),  # None in the middle
        ([1, 2, 3, 4, None], 3, None),  # None at the end
        ([None, None, 3, 4, 5], 3, None),  # Multiple Nones at the beginning
        ([1, 2, None, None, 5], 3, None),  # Multiple Nones in the middle
        ([1, 2, 3, None, None], 3, None),  # Multiple Nones at the end
        ([None, 2, None, 4, None], 3, None),  # Alternating Nones
        # min_periods scenarios
        ([None, None, 3, 4, 5], 3, 1),  # min_periods less than length, with Nones
        ([None, None, 3, 4, 5], 3, 2),  # min_periods less than length, with Nones
        ([None, None, 3, 4, 5], 3, 3),  # min_periods equals length, with Nones
        ([1, None, 3, None, 5], 2, 1),  # Short window with Nones, min_periods = 1
        # Special value patterns
        ([0, 0, 0, 0, 0], 3, None),  # All zeros
        ([-1, -2, -3, -4, -5], 3, None),  # Negative values
        ([1.5, 2.5, 3.5, 4.5, 5.5], 3, None),  # Fractional values
        ([100, 200, 300, 400, 500], 3, None),  # Large values
        ([0.001, 0.002, 0.003, 0.004, 0.005], 3, None),  # Small values
        # Longer data series
        (list(range(1, 21)), 10, None),  # 20 values, window of 10
        (list(range(1, 21)), 10, 5),  # 20 values, window of 10, min_periods = 5
        # Patterns
        ([10, 10, 10, 20, 20, 20, 10, 10, 10], 3, None),  # Step function
        ([1, 10, 1, 10, 1, 10, 1, 10], 4, None),  # Oscillating values
        ([1, 10, 100, 1000, 10000], 3, None),  # Exponentially increasing
        ([10000, 1000, 100, 10, 1], 3, None),  # Exponentially decreasing
    ],
)
def test_sma_calculation_specific_cases(data: list[float | None], length: int, min_periods: int | None) -> None:
    # Calculate reference values using pandas_ta
    pta = ta.sma(pd.Series(data), length=length, min_periods=min_periods)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | SMAHandler(length=length, min_periods=min_periods))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data={data}, length={length}, min_periods={min_periods}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


def test_sma_with_all_nones() -> None:
    """Test that SMA returns all None values when input is all None."""
    data = [None, None, None, None, None]

    # Calculate values using our handler
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | SMAHandler(length=3))

    # All results should be None
    assert all(result is None for result in handler_result)


def test_sma_mathematical_properties() -> None:
    """Test that SMA follows expected mathematical properties."""
    # Test that SMA of constant values equals the constant
    constant_value = 42.0
    data = [constant_value] * 10

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result = list(provider | SMAHandler(length=5))

    # After enough values, SMA of constant should equal the constant
    threshold = 1e-10
    for result in handler_result[4:]:  # index 4 is when we have 5 values
        assert result is not None
        assert abs(result - constant_value) < threshold


def test_sma_min_periods_behavior() -> None:
    """Test specific behaviors related to min_periods parameter."""
    data = [None, None, 3, 4, 5]

    # With min_periods=1, we should get values as soon as we have any valid data
    provider1: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    result1 = list(provider1 | SMAHandler(length=3, min_periods=1))

    # We should get None for the first two positions, then values
    assert result1[0] is None
    assert result1[1] is None
    assert result1[2] is not None  # Only one valid value (3)
    assert result1[3] is not None  # Two valid values (3, 4)
    assert result1[4] is not None  # Three valid values (3, 4, 5)

    # With min_periods=2, we should get values when we have at least 2 valid values
    provider2: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    result2 = list(provider2 | SMAHandler(length=3, min_periods=2))

    assert result2[0] is None
    assert result2[1] is None
    assert result2[2] is None  # Only one valid value (3)
    assert result2[3] is not None  # Two valid values (3, 4)
    assert result2[4] is not None  # Three valid values (3, 4, 5)

    # With min_periods=3, we should get values when we have at least 3 valid values
    provider3: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    result3 = list(provider3 | SMAHandler(length=3, min_periods=3))

    assert result3[0] is None
    assert result3[1] is None
    assert result3[2] is None  # Only one valid value (3)
    assert result3[3] is None  # Only two valid values (3, 4)
    assert result3[4] is not None  # Three valid values (3, 4, 5)
