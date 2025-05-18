from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.wma_handler import WMAHandler
from tests.utils import safe_allclose


def to_pandas_series(data: list[float]) -> Any:
    return pd.Series(data)


def calculate_pandas_ta_wma(data: list[float], length: int, asc: bool) -> list[float | None]:
    series = to_pandas_series(data)
    wma_result = ta.wma(series, length=length, asc=asc)
    res: list[float | None] = wma_result.dropna().tolist()
    return res


def calculate_our_wma(data: Iterable[Optional[float]], length: int, asc: bool) -> list[float | None]:
    provider = SimpleDataProvider(data)
    handler = WMAHandler(length=length, asc=asc, source=provider)
    return [x for x in handler if x is not None]


def test_wma_basic() -> None:
    data: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    length = 5
    asc = True

    pandas_result = calculate_pandas_ta_wma(data, length, asc)
    our_result = calculate_our_wma(data, length, asc)

    assert len(pandas_result) == len(our_result)
    assert safe_allclose(pandas_result, our_result)


@pytest.mark.parametrize("length,asc", [(3, True), (5, True), (10, True), (8, True)])
def test_wma_parametrized(length: int, asc: bool) -> None:
    data: list[float] = [float(i) for i in range(1, 20)]

    pandas_result = calculate_pandas_ta_wma(data, length, asc)
    our_result = calculate_our_wma(data, length, asc)

    assert len(pandas_result) == len(our_result)
    assert safe_allclose(pandas_result, our_result)


@given(
    data=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=15, max_size=100
    ),
    length=st.integers(min_value=2, max_value=10),
)
def test_wma_property_based(data: list[float], length: int) -> None:
    pandas_result = calculate_pandas_ta_wma(data, length, True)
    our_result = calculate_our_wma(data, length, True)

    assert len(pandas_result) == len(our_result)
    assert safe_allclose(pandas_result, our_result)


def test_wma_with_none_values() -> None:
    data_with_nones: list[Optional[float]] = [1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0]
    length = 5
    asc = True

    pandas_data = data_with_nones
    series = pd.Series(pandas_data)
    pandas_result = ta.wma(series, length=length, asc=asc)
    pandas_result_list = pandas_result.dropna().tolist()

    our_result = calculate_our_wma(data_with_nones, length, asc)

    assert len(pandas_result_list) == len(our_result)
    assert safe_allclose(pandas_result_list, our_result)


@pytest.mark.parametrize(
    "data,length,expected_result_length",
    [([1.0, 2.0, 3.0], 5, 0), ([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 5, 2)],
)
def test_wma_result_length(data: list[float], length: int, expected_result_length: int) -> None:
    our_result = calculate_our_wma(data, length, True)
    assert len(our_result) == expected_result_length


def test_wma_empty_input() -> None:
    data: list[float] = []
    our_result = calculate_our_wma(data, 5, True)
    assert len(our_result) == 0


def test_wma_default_parameters() -> None:
    data: list[float] = [float(i) for i in range(1, 15)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    default_handler = WMAHandler(source=provider)
    custom_handler = WMAHandler(length=10, asc=True, source=provider)

    default_result = [x for x in default_handler if x is not None]
    custom_result = [x for x in custom_handler if x is not None]

    assert len(default_result) == len(custom_result)
    for d, c in zip(default_result, custom_result):
        assert np.isclose(d, c)


def test_wma_desc() -> None:
    data: list[float] = [float(i) for i in range(1, 20)]

    expected_result: list[float | None] = [
        3.999999999999999,
        4.999999999999999,
        5.999999999999999,
        7.0,
        8.0,
        9.000000000000002,
        10.0,
        10.999999999999998,
        12.0,
        13.0,
    ]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler = WMAHandler(length=10, asc=False, source=provider)
    our_result: list[float | None] = [x for x in handler if x is not None]

    assert len(our_result) == len(expected_result)
    assert safe_allclose(our_result, expected_result)


def test_wma_specific_values() -> None:
    """Test with specific known values to verify correctness"""
    data: list[float] = [10.0, 20.0, 30.0, 40.0, 50.0]
    length = 3
    asc = True

    # Manual calculation for WMA with length 3, ascending:
    # WMA = (1*10 + 2*20 + 3*30) / (1+2+3) = (10 + 40 + 90) / 6 = 140 / 6 = 23.333...
    # Similarly for the next value: (1*20 + 2*30 + 3*40) / 6 = 33.333...
    expected_results: list[float | None] = [23.333333333333332, 33.333333333333336, 43.333333333333336]

    our_result = calculate_our_wma(data, length, asc)

    assert len(our_result) == len(expected_results)
    assert safe_allclose(our_result, expected_results)


def test_wma_edge_case_length_one() -> None:
    """Test with length=1, which should simply return the input values"""
    data: list[float | None] = [10.0, 20.0, 30.0, 40.0, 50.0]
    length = 1

    our_result = calculate_our_wma(data, length, True)

    # With length=1, the result should match the input
    assert len(our_result) == len(data)
    assert safe_allclose(our_result, data)
