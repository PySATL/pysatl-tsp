import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.ohlc4_handler import Ohlc4Handler
from tests.utils import safe_allclose


@given(
    data=st.lists(
        st.tuples(
            st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        ),
        min_size=1,
        max_size=50,
    ),
)
def test_ohlc4_calculation_property_based(
    data: list[tuple[float | None, float | None, float | None, float | None]],
) -> None:
    # Split the data into separate O, H, L, C lists for pandas_ta
    opens = [o for o, _, _, _ in data]
    highs = [h for _, h, _, _ in data]
    lows = [x for _, _, x, _ in data]
    closes = [c for _, _, _, c in data]

    # Calculate reference values using pandas_ta
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    pta = ta.ohlc4(open_=df["open"], high=df["high"], low=df["low"], close=df["close"])
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[tuple[float | None, float | None, float | None, float | None]] = SimpleDataProvider(
        data
    )
    handler_result: list[float | None] = list(provider | Ohlc4Handler())

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data: data={data}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data",
    [
        [
            (10, 12, 9, 11),  # Normal OHLC data point
            (11, 13, 10, 12),
            (12, 14, 11, 13),
        ],  # Normal data without any None values
        [
            (10, 12, 9, 11),
            (None, 13, 10, 12),  # None in Open
            (12, 14, 11, 13),
        ],  # Data with None in Open
        [
            (10, 12, 9, 11),
            (11, None, 10, 12),  # None in High
            (12, 14, 11, 13),
        ],  # Data with None in High
        [
            (10, 12, 9, 11),
            (11, 13, None, 12),  # None in Low
            (12, 14, 11, 13),
        ],  # Data with None in Low
        [
            (10, 12, 9, 11),
            (11, 13, 10, None),  # None in Close
            (12, 14, 11, 13),
        ],  # Data with None in Close
        [
            (10, 12, 9, 11),
            (None, None, None, None),  # All None values
            (12, 14, 11, 13),
        ],  # Data with a row of all None values
        [
            (1, 1, 1, 1),  # All values the same (OHLC4 should equal 1)
            (2, 2, 2, 2),  # All values the same (OHLC4 should equal 2)
        ],  # Constant data
        [
            (10, 20, 5, 15),  # Wide price range
            (15, 25, 10, 20),  # Wide price range
        ],  # Data with wide range between high and low
        [
            (10.5, 11.25, 9.75, 10.5),  # Fractional values
            (10.25, 11.5, 9.5, 10.75),  # Fractional values
        ],  # Data with fractional values
        [
            (-10, -5, -15, -8),  # Negative values
            (-8, -3, -12, -6),  # Negative values
        ],  # Data with negative values
        [
            (0, 0, 0, 0),  # Zero values
        ],  # Data with all zeros
        [
            (1000000, 1000010, 999990, 1000005),  # Very large values
        ],  # Large magnitude values
        [
            (0.00001, 0.00002, 0.00001, 0.000015),  # Very small values
        ],  # Small magnitude values
    ],
)
def test_ohlc4_calculation_specific_cases(
    data: list[tuple[float | None, float | None, float | None, float | None]],
) -> None:
    # Split the data into separate O, H, L, C lists for pandas_ta
    opens = [o for o, _, _, _ in data]
    highs = [h for _, h, _, _ in data]
    lows = [x for _, _, x, _ in data]
    closes = [c for _, _, _, c in data]

    # Calculate reference values using pandas_ta
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    pta = ta.ohlc4(open_=df["open"], high=df["high"], low=df["low"], close=df["close"])
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    # Calculate values using our handler
    provider: SimpleDataProvider[tuple[float | None, float | None, float | None, float | None]] = SimpleDataProvider(
        data
    )
    handler_result: list[float | None] = list(provider | Ohlc4Handler())

    # Check result lengths
    assert len(pta_result) == len(handler_result), f"Result lengths do not match for data: data={data}"

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result)
