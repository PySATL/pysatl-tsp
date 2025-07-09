import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.rma_handler import RMAHandler
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
    length=st.integers(min_value=2, max_value=20),
)
def test_rma_calculation_property_based(data: list[float | None], length: int) -> None:
    if length > len(data):
        return

    # If all values are None, no meaningful test can be performed
    if all(x is None for x in data):
        data = [*data, 1.0]  # Add at least one non-None value

    # Calculate reference values using pandas_ta
    pta = ta.rma(pd.Series(data), length=length)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | RMAHandler(length=length))

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data={data}, length={length}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data, length",
    [
        # Basic scenarios
        ([1, 2, 3, 4, 5], 3),  # Simple case
        ([1, 2, 3, 4, 5], 5),  # Length equals data length
        ([1, 2, 3, 4, 5], 1),  # Length = 1
        # None handling scenarios
        ([None, 2, 3, 4, 5], 3),  # None at the beginning
        ([1, 2, 3, None, 5], 3),  # None in the middle
        ([1, 2, 3, 4, None], 3),  # None at the end
        ([None, None, 3, 4, 5], 3),  # Multiple Nones at the beginning
        ([1, 2, None, None, 5], 3),  # Multiple Nones in the middle
        ([1, 2, 3, None, None], 3),  # Multiple Nones at the end
        ([None, 2, None, 4, None], 3),  # Alternating Nones
        # Special value patterns
        ([0, 0, 0, 0, 0], 3),  # All zeros
        ([-1, -2, -3, -4, -5], 3),  # Negative values
        ([1.5, 2.5, 3.5, 4.5, 5.5], 3),  # Fractional values
        ([100, 200, 300, 400, 500], 3),  # Large values
        ([0.001, 0.002, 0.003, 0.004, 0.005], 3),  # Small values
        # Longer data series
        (list(range(1, 21)), 10),  # 20 values, window of 10
        # Patterns
        ([10, 10, 10, 20, 20, 20, 10, 10, 10], 3),  # Step function
        ([1, 10, 1, 10, 1, 10, 1, 10], 4),  # Oscillating values
        ([1, 10, 100, 1000, 10000], 3),  # Exponentially increasing
        ([10000, 1000, 100, 10, 1], 3),  # Exponentially decreasing
        # Specific for RMA
        ([5, 5, 5, 5, 5], 5),  # Constant series
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5),  # Linear series
    ],
)
def test_rma_calculation_specific_cases(data: list[float | None], length: int) -> None:
    # Calculate reference values using pandas_ta
    pta = ta.rma(pd.Series(data), length=length)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | RMAHandler(length=length))

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data={data}, length={length}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


def test_rma_with_all_nones() -> None:
    """Test that RMA returns all None values when input is all None."""
    data = [None, None, None, None, None]

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | RMAHandler(length=3))

    # All results should be None
    assert all(result is None for result in handler_result)


def test_rma_mathematical_properties() -> None:
    """Test that RMA follows expected mathematical properties."""
    # Test that RMA of constant values equals the constant
    constant_value = 42.0
    data = [constant_value] * 15

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result = list(provider | RMAHandler(length=5))

    threshold = 1e-10
    # After enough values, RMA of constant should equal the constant
    for result in handler_result[4:]:  # Skip initial values before window is filled
        if result is not None:
            assert abs(result - constant_value) < threshold


def test_rma_alpha_calculation() -> None:
    """Test that the alpha value is calculated correctly as 1/length."""
    length = 5
    handler = RMAHandler(length=length)
    assert handler.alpha == 1 / length


def test_rma_requires_source() -> None:
    """Test that RMAHandler raises an error if no source is set."""
    handler = RMAHandler(length=5)

    # Attempting to iterate without a source should raise ValueError
    with pytest.raises(ValueError, match="Source is not set"):
        list(handler)


def test_rma_pipeline_chaining() -> None:
    """Test that RMAHandler can be chained in a pipeline."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    # Create a pipeline: data -> RMA -> SMA
    pipeline_result = list(provider | RMAHandler(length=5) | SMAHandler(length=3))

    # Create the same pipeline step by step for comparison
    step1_result = list(provider | RMAHandler(length=5))
    step2_provider = SimpleDataProvider(step1_result)
    step2_result = list(step2_provider | SMAHandler(length=3))

    # Results should match
    assert safe_allclose(pipeline_result, step2_result)


def test_rma_source_constructor() -> None:
    """Test that RMAHandler can use a source provided in constructor."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    # Create handler with source in constructor
    handler = RMAHandler(length=5, source=provider)
    constructor_result = list(handler)

    # Create handler with pipeline syntax
    pipeline_result = list(SimpleDataProvider[float | None](data) | RMAHandler(length=5))

    # Results should match
    assert safe_allclose(constructor_result, pipeline_result)
