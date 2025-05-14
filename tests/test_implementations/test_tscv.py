from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations import TimeSeriesCrossValidator


def test_basic_functionality() -> None:
    data = list(range(20))
    provider = SimpleDataProvider(data)

    min_train_size = 10
    val_size = 2
    cv = TimeSeriesCrossValidator(min_train_size=min_train_size, val_size=val_size, source=provider)

    splits = list(cv)

    assert len(splits) == (len(data) - min_train_size) // val_size

    # Check the first split
    train1, val1 = splits[0]
    assert len(train1) == min_train_size
    assert len(val1) == val_size
    assert list(train1.values) == list(range(10))
    assert list(val1.values) == [10, 11]

    # Check the last split
    train_last, val_last = splits[-1]
    assert len(train_last) == len(data) - val_size
    assert len(val_last) == val_size
    assert list(train_last.values) == list(range(18))
    assert list(val_last.values) == [18, 19]


def test_raises_error_when_source_not_set() -> None:
    cv: TimeSeriesCrossValidator[Any] = TimeSeriesCrossValidator(min_train_size=10, val_size=2)

    with pytest.raises(ValueError, match="Source is not set"):
        next(iter(cv))


@given(
    min_train_size=st.integers(min_value=1, max_value=20),
    val_size=st.integers(min_value=1, max_value=10),
    extra_data_points=st.integers(min_value=0, max_value=100),
)
def test_property_based(min_train_size: int, val_size: int, extra_data_points: int) -> None:
    total_data_size = min_train_size + val_size + extra_data_points
    data = list(range(total_data_size))

    provider = SimpleDataProvider(data)
    cv = TimeSeriesCrossValidator(min_train_size=min_train_size, val_size=val_size, source=provider)

    splits = list(cv)

    expected_splits = 1 + extra_data_points // val_size
    assert len(splits) == expected_splits

    for i, (train, val) in enumerate(splits):
        assert len(train) == min_train_size + i * val_size
        assert len(val) == val_size

        expected_val_start = min_train_size + i * val_size
        assert list(val.values) == data[expected_val_start : expected_val_start + val_size]
        assert list(train.values) == data[:expected_val_start]


@given(data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=15, max_size=50))
def test_window_indices_preserved(data: list[float]) -> None:
    provider = SimpleDataProvider(data)
    cv = TimeSeriesCrossValidator(min_train_size=10, val_size=5, source=provider)

    for train, val in cv:
        assert list(train.indices) == list(range(len(train)))
        assert list(val.indices) == list(range(len(train), len(train) + len(val)))
        assert train.indices[-1] + 1 == val.indices[0]


def test_empty_data() -> None:
    provider: SimpleDataProvider[Any] = SimpleDataProvider([])
    cv = TimeSeriesCrossValidator(min_train_size=5, val_size=2, source=provider)

    assert list(cv) == []


def test_insufficient_data() -> None:
    provider = SimpleDataProvider(list(range(9)))
    cv = TimeSeriesCrossValidator(min_train_size=10, val_size=2, source=provider)

    assert list(cv) == []


def test_exact_data_for_one_split() -> None:
    provider = SimpleDataProvider(list(range(12)))
    min_train_size = 10
    val_size = 2
    cv = TimeSeriesCrossValidator(min_train_size=min_train_size, val_size=val_size, source=provider)

    splits = list(cv)
    assert len(splits) == 1

    train, val = splits[0]
    assert len(train) == min_train_size
    assert len(val) == val_size
    assert list(train.values) == list(range(10))
    assert list(val.values) == [10, 11]
