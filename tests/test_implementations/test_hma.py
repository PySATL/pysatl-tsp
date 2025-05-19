import pandas as pd
import pandas_ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.hma_handler import HMAHandler
from tests.utils import safe_allclose


@given(
    data=st.lists(
        st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=2, max_value=20),  # HMA requires length >= 2
)
def test_hma_calculation_with_none_property_based(data: list[float | None], length: int) -> None:
    if length > len(data):
        return

    # Ensure we have at least one non-None value
    if all(x is None for x in data):
        data = [*data, 1.0]

    # Calculate reference values using pandas_ta
    pta = pandas_ta.hma(pd.Series(data), length=length)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | HMAHandler(length=length))

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data: data={data}, length={length}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data, length",
    [
        ([1, 3, 5, 4, 2], 2),  # Normal data
        ([1, None, 5, 4, 2], 2),  # None in the middle
        ([1, 3, 5, None, None], 2),  # None at the end
        ([None, None, 5, 4, 2], 2),  # None at the beginning
        ([1, 1, None, 1, 1], 3),  # Single None value
        ([1, 2, 3, 4, None], 3),  # None at the end, length 3
        ([None, 4, 3, 2, 1], 3),  # None at the beginning, length 3
        ([1, None, 1, None, 1], 2),  # Alternating None values
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),  # Long sequence without None
        # HMA-specific test cases
        ([1] * 10, 4),  # Constant data
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2),  # Length 2
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 9),  # Length close to data length
        ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 5),  # Descending data
        ([5, 5, 5, 10, 10, 10, 5, 5, 5], 3),  # Step function
        ([1.5, 2.5, 3.5, 4.5, 5.5], 4),  # Fractional values
    ],
)
def test_hma_calculation_with_none_specific_cases(data: list[float | None], length: int) -> None:
    pta = pandas_ta.hma(pd.Series(data), length)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | HMAHandler(length=length))

    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data: data={data}, length={length}"

    assert safe_allclose(pta_result, handler_result)
