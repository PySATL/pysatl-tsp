import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.pwma_handler import PWMAHandler
from tests.utils import safe_allclose


@given(
    data=st.lists(
        st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=1, max_value=20),
    asc=st.booleans(),
)
def test_pwma_calculation_with_none_property_based(data: list[float | None], length: int, asc: bool) -> None:
    if length > len(data):
        return

    # Ensure we have at least one non-None value
    if all(x is None for x in data):
        data = [*data, 1.0]

    # Calculate reference values using pandas_ta
    pta = ta.pwma(pd.Series(data), length=length, asc=asc)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | PWMAHandler(length=length, asc=asc))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data: data={data}, length={length}, asc={asc}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data, length, asc",
    [
        ([1, 3, 5, 4, 2], 2, True),  # Normal data, ascending weights
        ([1, 3, 5, 4, 2], 2, False),  # Normal data, descending weights
        ([1, None, 5, 4, 2], 2, True),  # None in the middle, ascending weights
        ([1, 3, 5, None, None], 2, True),  # None at the end, ascending weights
        ([None, None, 5, 4, 2], 2, True),  # None at the beginning, ascending weights
        ([1, 1, None, 1, 1], 3, True),  # Single None value, ascending weights
        ([1, 2, 3, 4, None], 3, True),  # None at the end, length 3, ascending weights
        ([None, 4, 3, 2, 1], 3, False),  # None at the beginning, length 3, descending weights
        ([1, None, 1, None, 1], 2, True),  # Alternating None values, ascending weights
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, True),  # Long sequence without None, ascending weights
        # PWMA-specific test cases
        ([1] * 10, 4, True),  # Constant data, ascending weights
        ([1] * 10, 4, False),  # Constant data, descending weights
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, True),  # Length 2, ascending weights
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 9, True),  # Length close to data length, ascending weights
        ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 5, False),  # Descending data, descending weights
        ([5, 5, 5, 10, 10, 10, 5, 5, 5], 3, True),  # Step function, ascending weights
        ([1.5, 2.5, 3.5, 4.5, 5.5], 4, True),  # Fractional values, ascending weights
        ([-1, -2, -3, -4, -5], 3, True),  # Negative values, ascending weights
        ([0, 0, 0, 0, 0], 3, True),  # Zero values, ascending weights
        # Test larger pascal triangle
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 10, True),  # Length 10 pascal weights
        # Test edge cases for length
        ([1, 2, 3], 1, True),  # Length 1
        ([1, 2, 3, 4, 5], 5, True),  # Length equals data length
    ],
)
def test_pwma_calculation_with_none_specific_cases(data: list[float | None], length: int, asc: bool) -> None:
    # Calculate reference values using pandas_ta
    pta = ta.pwma(pd.Series(data), length=length, asc=asc)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | PWMAHandler(length=length, asc=asc))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data: data={data}, length={length}, asc={asc}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize("length", [5, 10, 15, 20])
def test_pwma_pascal_triangle_weights(length: int) -> None:
    """Test that the Pascal triangle weights calculation is correct."""
    # Create the handler and get its weights
    handler = PWMAHandler(length=length)
    weights = handler.weights
    threshold: float = 1e-10

    # Check that the weights sum to approximately 1
    assert abs(sum(weights) - 1.0) < threshold

    # Check that the number of weights is correct
    assert len(weights) == length

    # For length 5, verify the weights match the expected values
    five: int = 5
    if length == five:
        # Pascal triangle row 4: [1, 4, 6, 4, 1], normalized
        expected = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
        for w, e in zip(weights, expected):
            assert abs(w - e) < threshold
