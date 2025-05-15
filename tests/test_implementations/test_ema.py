import pandas as pd
import pandas_ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.ema_handler import EMAHandler
from tests.utils import safe_allclose


@given(
    data=st.lists(
        st.one_of(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=1, max_value=20),
    sma=st.booleans(),
    adjust=st.booleans(),
)
def test_ema_calculation_with_none_property_based(data: list[float], length: int, sma: bool, adjust: bool) -> None:
    if length > len(data):
        return

    if all(x is None for x in data):
        data = [*data, 1.0]

    pta = pandas_ta.ema(pd.Series(data), length, talib=None, adjust=adjust, sma=sma)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | EMAHandler(length=length, adjust=adjust, sma=sma))

    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data: data={data}, length={length}, sma={sma}, adjust={adjust}"
    )

    assert safe_allclose(pta_result, handler_result)


@pytest.mark.parametrize(
    "data, length, sma, adjust, expected_equal",
    [
        ([1, 3, 5, 4, 2], 2, True, True, True),
        ([1, None, 5, 4, 2], 2, True, True, True),
        ([1, 3, 5, None, None], 2, True, False, True),
        ([None, None, 5, 4, 2], 2, False, True, True),
        ([1, 1, None, 1, 1], 3, True, True, True),
        ([1, 2, 3, 4, None], 3, True, True, True),
        ([None, 4, 3, 2, 1], 3, True, True, True),
        ([1, None, 1, None, 1], 2, True, True, True),
    ],
)
def test_ema_calculation_with_none_specific_cases(
    data: list[float | None], length: int, sma: bool, adjust: bool, expected_equal: bool
) -> None:
    pta = pandas_ta.ema(pd.Series(data), length, talib=None, adjust=adjust, sma=sma)
    pta_result: list[float | None] = [elem if not pd.isna(elem) else None for elem in list(pta)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | EMAHandler(length=length, adjust=adjust, sma=sma))

    if expected_equal:
        assert len(pta_result) == len(handler_result), (
            f"Result lengths do not match for data: data={data}, length={length}, sma={sma}, adjust={adjust}"
        )

        assert safe_allclose(pta_result, handler_result)
