from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.julia import JuliaSMAHandler
from pysatl_tsp.implementations.processor.julia import _base as julia_base
from pysatl_tsp.implementations.processor.julia._bridge import get_backend
from pysatl_tsp.implementations.processor.sma_handler import SMAHandler
from tests.utils import JuliaBackendStub, safe_allclose


@pytest.fixture(scope="module")
def julia_backend() -> Any:
    """Provide the Julia TSPBackend module or skip if Julia is unavailable.

    The Julia runtime is an optional heavy dependency: integration tests are
    skipped (not failed) when juliacall is not installed or the Julia binary
    cannot be initialized in the current environment.

    :return: Handle to the Julia backend module
    """
    try:
        return get_backend()
    except Exception as exc:
        # juliacall may raise not only ImportError but also e.g. network errors
        # while bootstrapping the Julia binary, so we catch broadly.
        pytest.skip(f"Julia runtime is not available: {exc}")


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> JuliaBackendStub:
    """Replace the Julia backend with a pure-Python double.

    :param monkeypatch: pytest monkeypatch fixture
    :return: The stub backend used by every Julia handler in the test
    """
    backend = JuliaBackendStub()
    monkeypatch.setattr(julia_base, "get_backend", lambda: backend)
    return backend


# ---------------------------------------------------------------------------
# Unit tests: no Julia runtime required
# ---------------------------------------------------------------------------


def test_julia_sma_pipeline_plumbing_with_stub_backend(stub: JuliaBackendStub) -> None:
    provider: SimpleDataProvider[float | None] = SimpleDataProvider([1.0, None, 3.0, 4.0])

    result = list(provider | JuliaSMAHandler(length=3, min_periods=2))

    assert result[:2] == [None, None]
    assert safe_allclose(result, [None, None, 2.0, 3.5])


def test_julia_sma_min_periods_defaults_to_length() -> None:
    length = 4
    handler = JuliaSMAHandler(length=length)
    assert handler.min_periods == length


def test_julia_sma_backend_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the handler and building a pipeline must not touch Julia."""

    def _fail() -> Any:
        raise AssertionError("Julia backend must not be initialized while building a pipeline")

    monkeypatch.setattr(julia_base, "get_backend", _fail)

    provider: SimpleDataProvider[float | None] = SimpleDataProvider([1.0, 2.0, 3.0])
    _ = provider | JuliaSMAHandler(length=3)


@given(
    data=st.lists(
        st.one_of(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            st.none(),
        ),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=1, max_value=20),
    min_periods=st.integers(min_value=1, max_value=20) | st.none(),
)
def test_julia_sma_stub_matches_python_sma(data: list[float | None], length: int, min_periods: int | None) -> None:
    """The backend contract must match SMAHandler semantics exactly."""
    if length > len(data):
        return
    effective_min_periods = min(min_periods, length) if min_periods is not None else length

    with patch.object(julia_base, "get_backend", JuliaBackendStub):
        provider_a: SimpleDataProvider[float | None] = SimpleDataProvider(data)
        stub_result = list(provider_a | JuliaSMAHandler(length=length, min_periods=effective_min_periods))

    provider_b: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    python_result = list(provider_b | SMAHandler(length=length, min_periods=effective_min_periods))

    assert len(stub_result) == len(python_result)
    assert [x is None for x in stub_result] == [x is None for x in python_result]
    assert safe_allclose(stub_result, python_result)


# ---------------------------------------------------------------------------
# Integration tests: require a working Julia runtime, otherwise skipped
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    data=st.lists(
        st.one_of(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            st.none(),
        ),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=1, max_value=20),
    min_periods=st.integers(min_value=1, max_value=20) | st.none(),
)
def test_julia_sma_matches_python_sma_property_based(
    julia_backend: Any, data: list[float | None], length: int, min_periods: int | None
) -> None:
    if length > len(data):
        return
    effective_min_periods = min(min_periods, length) if min_periods is not None else length

    provider_a: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    julia_result = list(provider_a | JuliaSMAHandler(length=length, min_periods=effective_min_periods))

    provider_b: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    python_result = list(provider_b | SMAHandler(length=length, min_periods=effective_min_periods))

    assert len(julia_result) == len(python_result), (
        f"Result lengths do not match for data={data}, length={length}, min_periods={effective_min_periods}"
    )
    assert [x is None for x in julia_result] == [x is None for x in python_result]
    assert safe_allclose(julia_result, python_result)


@pytest.mark.parametrize(
    "data, length, min_periods, expected",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3, None, [None, None, 2.0, 3.0, 4.0]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3, 1, [1.0, 1.5, 2.0, 3.0, 4.0]),
        ([1.0, None, 3.0, 4.0], 3, 2, [None, None, 2.0, 3.5]),
        ([1.0, 2.0, 3.0], 1, None, [1.0, 2.0, 3.0]),
        ([None, None, None], 2, 1, [None, None, None]),
    ],
)
def test_julia_sma_known_values(
    julia_backend: Any,
    data: list[float | None],
    length: int,
    min_periods: int | None,
    expected: list[float | None],
) -> None:
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    result = list(provider | JuliaSMAHandler(length=length, min_periods=min_periods))

    assert [x is None for x in result] == [x is None for x in expected]
    assert safe_allclose(result, expected)


def test_julia_backend_chain_works_directly(julia_backend: Any) -> None:
    """Sanity check of the Julia chain itself, bypassing the handlers."""
    chain = julia_backend.chain_of(julia_backend.sma(3, 2))

    assert julia_backend.step_chain(chain, 1.0, False)[1] is True
    value, is_missing = julia_backend.step_chain(chain, 3.0, False)
    assert is_missing is False
    assert value == pytest.approx(2.0)
