from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.fwma_handler import FWMAHandler
from tests.utils import safe_allclose


def to_pandas_series(data: list[float]) -> Any:
    return pd.Series(data)


def calculate_pandas_ta_fwma(data: list[float], length: int, asc: bool) -> list[float | None]:
    series = to_pandas_series(data)
    fwma_result = ta.fwma(series, length=length, asc=asc)
    res: list[float | None] = fwma_result.dropna().tolist()
    return res


def calculate_our_fwma(data: Iterable[Optional[float]], length: int, asc: bool) -> list[float | None]:
    provider = SimpleDataProvider(data)
    handler = FWMAHandler(length=length, asc=asc, source=provider)
    return [x for x in handler if x is not None]


def test_fwma_basic() -> None:
    data: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    length = 5
    asc = True

    pandas_result = calculate_pandas_ta_fwma(data, length, asc)
    our_result = calculate_our_fwma(data, length, asc)

    assert len(pandas_result) == len(our_result)
    assert safe_allclose(pandas_result, our_result)


@pytest.mark.parametrize("length,asc", [(3, True), (5, True), (10, True), (8, True)])
def test_fwma_parametrized(length: int, asc: bool) -> None:
    data: list[float] = [float(i) for i in range(1, 20)]

    pandas_result = calculate_pandas_ta_fwma(data, length, asc)
    our_result = calculate_our_fwma(data, length, asc)

    assert len(pandas_result) == len(our_result)
    assert safe_allclose(pandas_result, our_result)


@given(
    data=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=15, max_size=100
    ),
    length=st.integers(min_value=2, max_value=10),
)
def test_fwma_property_based(data: list[float], length: int) -> None:
    pandas_result = calculate_pandas_ta_fwma(data, length, True)
    our_result = calculate_our_fwma(data, length, True)

    assert len(pandas_result) == len(our_result)
    assert safe_allclose(pandas_result, our_result)


def test_fwma_with_none_values() -> None:
    data_with_nones: list[Optional[float]] = [1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0]
    length = 5
    asc = True

    pandas_data = data_with_nones
    series = pd.Series(pandas_data)
    pandas_result = ta.fwma(series, length=length, asc=asc)
    pandas_result_list = pandas_result.dropna().tolist()

    our_result = calculate_our_fwma(data_with_nones, length, asc)

    assert len(pandas_result_list) == len(our_result)
    assert safe_allclose(pandas_result_list, our_result)


@pytest.mark.parametrize(
    "data,length,expected_result_length",
    [([1.0, 2.0, 3.0], 5, 0), ([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 5, 2)],
)
def test_fwma_result_length(data: list[float], length: int, expected_result_length: int) -> None:
    our_result = calculate_our_fwma(data, length, True)
    assert len(our_result) == expected_result_length


def test_fwma_empty_input() -> None:
    data: list[float] = []
    our_result = calculate_our_fwma(data, 5, True)
    assert len(our_result) == 0


def test_fwma_default_parameters() -> None:
    data: list[float] = [float(i) for i in range(1, 15)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    default_handler = FWMAHandler(source=provider)
    custom_handler = FWMAHandler(length=10, asc=True, source=provider)

    default_result = [x for x in default_handler if x is not None]
    custom_result = [x for x in custom_handler if x is not None]

    assert len(default_result) == len(custom_result)
    for d, c in zip(default_result, custom_result):
        assert np.isclose(d, c)


def test_fwma_desc() -> None:
    data: list[float] = [float(i) for i in range(1, 20)]

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler = FWMAHandler(length=10, asc=False, source=provider)

    default_result = [x for x in handler if x is not None]
    correct_result = [
        2.545454545454546,
        3.545454545454546,
        4.545454545454546,
        5.545454545454547,
        6.545454545454546,
        7.545454545454546,
        8.545454545454547,
        9.545454545454545,
        10.545454545454547,
        11.545454545454547,
    ]

    assert len(default_result) == len(correct_result)
    for d, c in zip(default_result, correct_result):
        assert np.isclose(d, c)
