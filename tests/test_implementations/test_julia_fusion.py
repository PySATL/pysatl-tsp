from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.core.processor import MappingHandler
from pysatl_tsp.implementations.processor.julia import (
    JuliaMedianFilterHandler,
    JuliaSMAHandler,
)
from pysatl_tsp.implementations.processor.julia import _base as julia_base
from tests.utils import JuliaBackendStub, safe_allclose


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> JuliaBackendStub:
    """Replace the Julia backend with a pure-Python double.

    :param monkeypatch: pytest monkeypatch fixture
    :return: The stub backend used by every Julia handler in the test
    """
    backend = JuliaBackendStub()
    monkeypatch.setattr(julia_base, "get_backend", lambda: backend)
    return backend


DATA: list[float | None] = [1.0, 5.0, 2.0, 100.0, 3.0, None, 4.0, 2.0, 50.0, 1.0]


def _identity() -> MappingHandler[float | None, float | None]:
    """Build a non-Julia handler that passes values through unchanged.

    Placed between two Julia handlers it breaks the run of Julia handlers and
    therefore disables fusion, which is what makes it useful as a reference.

    :return: A transparent mapping handler
    """
    return MappingHandler(map_func=lambda value: value)


def test_adjacent_julia_handlers_are_fused_into_one_chain(stub: JuliaBackendStub) -> None:
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(DATA)
    pipeline = provider | JuliaSMAHandler(length=3) | JuliaMedianFilterHandler(length=3)

    list(pipeline)

    assert [len(chain.ops) for chain in stub.chains] == [2]


def test_three_adjacent_julia_handlers_are_fused_into_one_chain(stub: JuliaBackendStub) -> None:
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(DATA)
    pipeline = provider | JuliaSMAHandler(length=3) | JuliaMedianFilterHandler(length=3) | JuliaSMAHandler(length=2)

    list(pipeline)

    assert [len(chain.ops) for chain in stub.chains] == [3]


def test_single_julia_handler_runs_as_chain_of_one(stub: JuliaBackendStub) -> None:
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(DATA)

    list(provider | JuliaSMAHandler(length=3))

    assert [len(chain.ops) for chain in stub.chains] == [1]


def test_non_julia_handler_interrupts_fusion(stub: JuliaBackendStub) -> None:
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(DATA)
    pipeline = provider | JuliaSMAHandler(length=3) | _identity() | JuliaMedianFilterHandler(length=3)

    list(pipeline)

    assert [len(chain.ops) for chain in stub.chains] == [1, 1]


def test_fusion_preserves_operation_order(stub: JuliaBackendStub) -> None:
    """Operations must reach Julia upstream-first, not in reverse."""
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(DATA)
    pipeline = provider | JuliaSMAHandler(length=4, min_periods=1) | JuliaMedianFilterHandler(length=2, min_periods=1)

    list(pipeline)

    lengths = [op.length for op in stub.chains[0].ops]
    assert lengths == [4, 2]


def test_missing_values_propagate_through_the_fused_chain(stub: JuliaBackendStub) -> None:
    """A missing intermediate result must advance the next window, not vanish."""
    data: list[float | None] = [1.0, 2.0, 3.0, 4.0]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    result = list(provider | JuliaSMAHandler(length=3) | JuliaMedianFilterHandler(length=2, min_periods=1))

    # SMA(3) yields None, None, 2.0, 3.0 -> median filter sees two missing
    # observations first and only then produces values.
    assert [value is None for value in result] == [True, True, False, False]
    assert safe_allclose(result, [None, None, 2.0, 2.5])


def test_julia_handler_without_source_raises(stub: JuliaBackendStub) -> None:
    with pytest.raises(ValueError, match="Source is not set"):
        list(JuliaSMAHandler(length=3))


def test_building_a_fused_pipeline_does_not_start_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime must stay untouched until the pipeline is actually consumed."""

    def _fail() -> Any:
        raise AssertionError("Julia backend must not be initialized while building a pipeline")

    monkeypatch.setattr(julia_base, "get_backend", _fail)

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(DATA)
    _ = provider | JuliaSMAHandler(length=3) | JuliaMedianFilterHandler(length=3)


@given(
    data=st.lists(
        st.one_of(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            st.none(),
        ),
        min_size=1,
        max_size=40,
    ),
    sma_length=st.integers(min_value=1, max_value=8),
    median_length=st.integers(min_value=1, max_value=8),
)
def test_fused_and_unfused_pipelines_agree(data: list[float | None], sma_length: int, median_length: int) -> None:
    """Fusion is an optimization: it must not change a single output value."""
    with patch.object(julia_base, "get_backend", JuliaBackendStub):
        fused_provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
        fused = list(
            fused_provider | JuliaSMAHandler(length=sma_length) | JuliaMedianFilterHandler(length=median_length)
        )

        unfused_provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
        unfused = list(
            unfused_provider
            | JuliaSMAHandler(length=sma_length)
            | _identity()
            | JuliaMedianFilterHandler(length=median_length)
        )

    assert [value is None for value in fused] == [value is None for value in unfused]
    assert safe_allclose(fused, unfused)
