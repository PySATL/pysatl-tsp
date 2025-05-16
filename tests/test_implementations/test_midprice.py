import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.midprice_handler import MidpriceHandler
from tests.utils import safe_allclose


@given(
    data=st.lists(
        st.tuples(
            st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        ),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=1, max_value=20),
)
def test_midprice_calculation_property_based(data: list[tuple[float | None, float | None]], length: int) -> None:
    if length > len(data):
        return

    # Ensure we have valid high/low data (high >= low)
    valid_data = []
    for high, low in data:
        if high is None or low is None:
            valid_data.append((high, low))
        else:
            # Ensure high >= low
            new_high = max(high, low)
            new_low = min(high, low)
            valid_data.append((new_high, new_low))

    data = valid_data

    # Split into separate high and low series for pandas_ta
    highs = [h for h, _ in data]
    lows = [x for _, x in data]

    # Calculate reference values using pandas_ta
    pta = ta.midprice(pd.Series(highs), pd.Series(lows), length=length)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[tuple[float | None, float | None]] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | MidpriceHandler(length=length))

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data: data={data}, length={length}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data, length",
    [
        ([(10, 5), (12, 6), (15, 8), (11, 7), (9, 4)], 2),  # Normal data
        ([(10, 5), (None, 6), (15, 8), (11, 7), (9, 4)], 2),  # None in high value
        ([(10, 5), (12, None), (15, 8), (11, 7), (9, 4)], 2),  # None in low value
        ([(10, 5), (None, None), (15, 8), (11, 7), (9, 4)], 2),  # None in both high and low
        ([(10, 5), (12, 6), (15, 8), (None, None), (None, None)], 2),  # None at the end
        ([(None, None), (None, None), (15, 8), (11, 7), (9, 4)], 2),  # None at the beginning
        ([(10, 10), (12, 12), (15, 15), (11, 11), (9, 9)], 3),  # High = Low (constant values)
        ([(10, 5), (20, 15), (30, 25), (40, 35), (50, 45)], 3),  # Ascending data
        ([(50, 45), (40, 35), (30, 25), (20, 15), (10, 5)], 3),  # Descending data
        (
            [(10, 5), (12, 6), (15, 8), (11, 7), (9, 4), (10, 5), (12, 6), (15, 8), (11, 7), (9, 4)],
            4,
        ),  # Repeating pattern
        ([(100, 90), (5, 1), (100, 90), (5, 1), (100, 90)], 3),  # Large oscillations
        ([(10.5, 9.5), (11.25, 10.75), (12.5, 11.5), (11.75, 10.25)], 2),  # Decimal values
        ([(-10, -15), (-8, -12), (-6, -9), (-4, -7), (-2, -5)], 3),  # Negative values
    ],
)
def test_midprice_calculation_specific_cases(data: list[tuple[float | None, float | None]], length: int) -> None:
    # Split into separate high and low series for pandas_ta
    highs = [h for h, _ in data]
    lows = [x for _, x in data]

    # Calculate reference values using pandas_ta
    pta = ta.midprice(pd.Series(highs), pd.Series(lows), length=length)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[tuple[float | None, float | None]] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | MidpriceHandler(length=length))

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data: data={data}, length={length}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)
