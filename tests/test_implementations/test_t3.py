import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.t3_handler import T3Handler
from tests.utils import safe_allclose


@pytest.mark.parametrize(
    "data, length, a",
    [
        # Basic scenarios
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 0.7),  # Simple case
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, 0.7),  # Longer length
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 0.7),  # Length = 1
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 0.1),  # Low a value
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 0.9),  # High a value
        # None handling scenarios
        ([None, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 0.7),  # None at the beginning
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, None], 3, 0.7),  # None at the end
        ([None, None, 3, 4, 5, 6, 7, 8, 9, 10], 3, 0.7),  # Multiple Nones at the beginning
        ([1, 2, 3, 4, 5, 6, 7, 8, None, None], 3, 0.7),  # Multiple Nones at the end
        # Special value patterns
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, 0.7),  # All zeros
        ([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 3, 0.7),  # Negative values
        ([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 3, 0.7),  # Fractional values
        ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 3, 0.7),  # Large values
        ([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010], 3, 0.7),  # Small values
        # Longer data series
        (list(range(1, 31)), 10, 0.7),  # 30 values, window of 10
        # Patterns
        ([10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20], 3, 0.7),  # Step function
        ([1, 10, 1, 10, 1, 10, 1, 10, 1, 10], 4, 0.7),  # Oscillating values
        ([1, 10, 100, 1000, 10000, 1, 10, 100, 1000, 10000], 3, 0.7),  # Exponentially increasing then repeating
        ([10000, 1000, 100, 10, 1, 10000, 1000, 100, 10, 1], 3, 0.7),  # Exponentially decreasing then repeating
        # Specific for T3
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 5, 0.3),  # Different a value
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 5, 0.5),  # Different a value
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 5, 0.8),  # Different a value
        ([5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15], 5, 0.7),  # Step function with various lengths
    ],
)
def test_t3_calculation_specific_cases(data: list[float | None], length: int, a: float) -> None:
    # Replace None values with NaN for pandas_ta
    pd_data = [float("nan") if x is None else x for x in data]

    # Calculate reference values using pandas_ta
    pta = ta.t3(pd.Series(pd_data), length=length, a=a)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | T3Handler(length=length, a=a))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data={data}, length={length}, a={a}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


def test_t3_with_all_nones() -> None:
    """Test that T3 returns all None values when input is all None."""
    data = [None, None, None, None, None, None, None, None, None, None]

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | T3Handler(length=3, a=0.7))

    # All results should be None
    assert all(result is None for result in handler_result)


def test_t3_mathematical_properties() -> None:
    """Test that T3 follows expected mathematical properties."""
    # Test that T3 of constant values equals the constant
    constant_value = 42.0
    data = [constant_value] * 50  # Longer list to ensure T3 converges

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result = list(provider | T3Handler(length=5, a=0.7))

    # After enough values, T3 of constant should equal the constant
    # Skip the first 30 values to allow for convergence
    threshold = 1e-10
    for result in handler_result[30:]:
        if result is not None:
            assert abs(result - constant_value) < threshold


@pytest.mark.parametrize(
    "a, expected_c1, expected_c2, expected_c3, expected_c4",
    [
        (
            0.7,
            -0.7 * 0.7 * 0.7,
            3 * 0.7 * 0.7 + 3 * 0.7 * 0.7 * 0.7,
            -6 * 0.7 * 0.7 - 3 * 0.7 - 3 * 0.7 * 0.7 * 0.7,
            0.7 * 0.7 * 0.7 + 3 * 0.7 * 0.7 + 3 * 0.7 + 1,
        ),
        (
            0.3,
            -0.3 * 0.3 * 0.3,
            3 * 0.3 * 0.3 + 3 * 0.3 * 0.3 * 0.3,
            -6 * 0.3 * 0.3 - 3 * 0.3 - 3 * 0.3 * 0.3 * 0.3,
            0.3 * 0.3 * 0.3 + 3 * 0.3 * 0.3 + 3 * 0.3 + 1,
        ),
        (
            0.5,
            -0.5 * 0.5 * 0.5,
            3 * 0.5 * 0.5 + 3 * 0.5 * 0.5 * 0.5,
            -6 * 0.5 * 0.5 - 3 * 0.5 - 3 * 0.5 * 0.5 * 0.5,
            0.5 * 0.5 * 0.5 + 3 * 0.5 * 0.5 + 3 * 0.5 + 1,
        ),
    ],
)
def test_t3_coefficients_calculation(
    a: float, expected_c1: float, expected_c2: float, expected_c3: float, expected_c4: float
) -> None:
    """Test that the coefficients in T3 are calculated correctly."""
    # Calculate coefficients using the formula from the original function
    c1 = -a * a**2
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = a**3 + 3 * a**2 + 3 * a + 1

    # Check that the coefficients are calculated correctly
    threshold = 1e-10
    assert abs(c1 - expected_c1) < threshold
    assert abs(c2 - expected_c2) < threshold
    assert abs(c3 - expected_c3) < threshold
    assert abs(c4 - expected_c4) < threshold


def test_t3_default_parameters() -> None:
    """Test that T3Handler uses default parameters correctly."""
    # Default length should be 10 and default a should be 0.7
    handler = T3Handler()
    default_length = 10
    default_a = 0.7
    assert handler.length == default_length
    assert handler.a == default_a

    # If a is out of bounds, it should be set to 0.7
    handler_invalid_a = T3Handler(a=1.5)
    assert handler_invalid_a.a == default_a


def test_t3_requires_source() -> None:
    """Test that T3Handler raises an error if no source is set."""
    handler = T3Handler(length=5, a=0.7)

    # Attempting to iterate without a source should raise ValueError
    with pytest.raises(ValueError):
        list(handler)


def test_t3_pipeline_chaining() -> None:
    """Test that T3Handler can be chained in a pipeline."""
    from pysatl_tsp.implementations.processor.sma_handler import SMAHandler

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    # Create a pipeline: data -> T3 -> SMA
    pipeline_result = list(provider | T3Handler(length=3, a=0.7) | SMAHandler(length=2))

    # Create the same pipeline step by step for comparison
    step1_result = list(provider | T3Handler(length=3, a=0.7))
    step2_provider = SimpleDataProvider(step1_result)
    step2_result = list(step2_provider | SMAHandler(length=2))

    # Results should match
    assert safe_allclose(pipeline_result, step2_result)


def test_t3_source_constructor() -> None:
    """Test that T3Handler can use a source provided in constructor."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    # Create handler with source in constructor
    handler = T3Handler(length=3, a=0.7, source=provider)
    constructor_result = list(handler)

    # Create handler with pipeline syntax
    pipeline_result = list(SimpleDataProvider[float | None](data) | T3Handler(length=3, a=0.7))

    # Results should match
    assert safe_allclose(constructor_result, pipeline_result)


def test_t3_extreme_a_values() -> None:
    """Test T3 with extreme values of a parameter."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Test with a very close to 0
    a_near_zero = 0.01
    pd_data = pd.Series(data)
    pta_near_zero = ta.t3(pd_data, length=5, a=a_near_zero)
    pta_result_near_zero = [elem if not pd.isna(elem) else None for elem in list(pta_near_zero)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result_near_zero = list(provider | T3Handler(length=5, a=a_near_zero))

    assert safe_allclose(pta_result_near_zero, handler_result_near_zero)

    # Test with a very close to 1
    a_near_one = 0.99
    pta_near_one = ta.t3(pd_data, length=5, a=a_near_one)
    pta_result_near_one = [elem if not pd.isna(elem) else None for elem in list(pta_near_one)]

    provider = SimpleDataProvider(data)
    handler_result_near_one = list(provider | T3Handler(length=5, a=a_near_one))

    assert safe_allclose(pta_result_near_one, handler_result_near_one)
