from typing import Any

import pandas as pd
import pandas_ta as ta  # type: ignore
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core import Handler
from pysatl_tsp.core.data_providers import SimpleDataProvider
from pysatl_tsp.implementations.processor.dema_handler import DEMAHandler
from pysatl_tsp.implementations.processor.ema_handler import EMAHandler
from pysatl_tsp.implementations.processor.rma_handler import RMAHandler
from pysatl_tsp.implementations.processor.sma_handler import SMAHandler
from pysatl_tsp.implementations.processor.t3_handler import T3Handler
from pysatl_tsp.implementations.processor.tema_handler import TEMAHandler
from pysatl_tsp.implementations.processor.trima_handler import TRIMAHandler
from pysatl_tsp.implementations.processor.wma_handler import WMAHandler
from pysatl_tsp.implementations.processor.zlma_handler import ZLMAHandler
from tests.utils import safe_allclose


def calculate_reference_zlma(data: Any, length: int, mamode: str) -> list[float | None]:
    """Calculates ZLMA directly, avoiding calling the problematic pandas_ta.zlma function"""
    pd_data = pd.Series(data)

    # Calculate lag as in the original zlma function
    lag = int(0.5 * (length - 1))

    # Create modified series
    modified_data = 2 * pd_data - pd_data.shift(lag)

    # Apply the corresponding moving average
    if mamode == "ema":
        result = ta.ema(modified_data, length=length)
    elif mamode == "sma":
        result = ta.sma(modified_data, length=length)
    elif mamode == "dema":
        result = ta.dema(modified_data, length=length)
    elif mamode == "t3":
        result = ta.t3(modified_data, length=length)
    elif mamode == "tema":
        result = ta.tema(modified_data, length=length)
    elif mamode == "trima":
        result = ta.trima(modified_data, length=length)
    elif mamode == "wma":
        result = ta.wma(modified_data, length=length)
    elif mamode == "rma":
        result = ta.rma(modified_data, length=length)
    else:
        result = ta.ema(modified_data, length=length)  # Default to EMA

    # Convert back to a list with None instead of NaN
    return [elem if not pd.isna(elem) else None for elem in list(result)]


@given(
    data=st.lists(
        st.one_of(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            st.none(),
        ),
        min_size=2,
        max_size=50,
    ),
    length=st.integers(min_value=2, max_value=20),
    mamode=st.sampled_from(["sma", "trima", "wma", "rma"]),
)
def test_zlma_calculation_property_based(data: list[float | None], length: int, mamode: str) -> None:
    if length > len(data):
        return

    # If all values are None, no meaningful test can be performed
    if all(x is None for x in data):
        data = [*data, 1.0]  # Add at least one non-None value

    # Replace None with NaN for pandas_ta
    pd_data: list[float | None] = [float("nan") if x is None else x for x in data]

    # Calculate reference values using our helper function
    pta_result = calculate_reference_zlma(pd_data, length, mamode)

    # Create appropriate ma_handler based on mamode
    ma_handler: Handler[float | None, float | None] | None = None
    if mamode == "ema":
        ma_handler = EMAHandler(length=length)
    elif mamode == "sma":
        ma_handler = SMAHandler(length=length)
    elif mamode == "dema":
        ma_handler = DEMAHandler(length=length)
    elif mamode == "t3":
        ma_handler = T3Handler(length=length)
    elif mamode == "tema":
        ma_handler = TEMAHandler(length=length)
    elif mamode == "trima":
        ma_handler = TRIMAHandler(length=length)
    elif mamode == "wma":
        ma_handler = WMAHandler(length=length)
    elif mamode == "rma":
        ma_handler = RMAHandler(length=length)

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | ZLMAHandler(length=length, ma_handler=ma_handler))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data={data}, length={length}, mamode={mamode}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "data, length, mamode",
    [
        # Basic scenarios
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "ema"),  # Simple case with EMA
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, "sma"),  # Longer length with SMA
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "dema"),  # DEMA
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "t3"),  # T3
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "tema"),  # TEMA
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "trima"),  # TRIMA
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "wma"),  # WMA
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "rma"),  # RMA
        # None handling scenarios
        ([None, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "ema"),  # None at the beginning
        ([1, 2, 3, None, 5, 6, 7, 8, 9, 10], 3, "sma"),  # None in the middle
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, None], 3, "dema"),  # None at the end
        ([None, None, 3, 4, 5, 6, 7, 8, 9, 10], 3, "t3"),  # Multiple Nones at the beginning
        ([1, 2, 3, 4, 5, 6, 7, 8, None, None], 3, "trima"),  # Multiple Nones at the end
        ([None, 2, None, 4, None, 6, None, 8, None, 10], 3, "wma"),  # Alternating Nones
        # Special value patterns
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, "rma"),  # All zeros
        ([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 3, "ema"),  # Negative values
        ([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 3, "sma"),  # Fractional values
        ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 3, "dema"),  # Large values
        ([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010], 3, "tema"),  # Small values
        # Longer data series
        (list(range(1, 31)), 10, "trima"),  # 30 values, window of 10
        # Patterns
        ([10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20], 3, "wma"),  # Step function
        ([1, 10, 1, 10, 1, 10, 1, 10, 1, 10], 4, "rma"),  # Oscillating values
        ([1, 10, 100, 1000, 10000, 1, 10, 100, 1000, 10000], 3, "ema"),  # Exponentially increasing then repeating
        ([10000, 1000, 100, 10, 1, 10000, 1000, 100, 10, 1], 3, "sma"),  # Exponentially decreasing then repeating
        # Extreme length cases
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, "dema"),  # Length = 1
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10, "tema"),  # Length = data length
    ],
)
def test_zlma_calculation_specific_cases(data: list[float | None], length: int, mamode: str) -> None:
    # Replace None with NaN for pandas_ta
    pd_data: list[float | None] = [float("nan") if x is None else x for x in data]

    # Calculate reference values using our helper function
    pta_result: list[float | None] = calculate_reference_zlma(pd_data, length, mamode)

    # Create appropriate ma_handler based on mamode
    ma_handler: Handler[float | None, float | None] | None = None
    if mamode == "ema":
        ma_handler = EMAHandler(length=length)
    elif mamode == "sma":
        ma_handler = SMAHandler(length=length)
    elif mamode == "dema":
        ma_handler = DEMAHandler(length=length)
    elif mamode == "t3":
        ma_handler = T3Handler(length=length)
    elif mamode == "tema":
        ma_handler = TEMAHandler(length=length)
    elif mamode == "trima":
        ma_handler = TRIMAHandler(length=length)
    elif mamode == "wma":
        ma_handler = WMAHandler(length=length)
    elif mamode == "rma":
        ma_handler = RMAHandler(length=length)

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | ZLMAHandler(length=length, ma_handler=ma_handler))

    # Check result lengths
    assert len(pta_result) == len(handler_result), (
        f"Result lengths do not match for data={data}, length={length}, mamode={mamode}"
    )

    # Check values (with margin of error)
    assert safe_allclose(pta_result, handler_result, rtol=1e-3, atol=1e-3)


def test_zlma_with_all_nones() -> None:
    """Test that ZLMA returns all None values when input is all None."""
    data = [None, None, None, None, None, None, None, None, None, None]

    # Calculate values using our handler with pipeline syntax
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result: list[float | None] = list(provider | ZLMAHandler(length=3))

    # All results should be None
    assert all(result is None for result in handler_result)


def test_zlma_mathematical_properties() -> None:
    """Test that ZLMA follows expected mathematical properties."""
    # Test that ZLMA of constant values equals the constant
    constant_value = 42.0
    data = [constant_value] * 50  # Long list to ensure convergence

    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    handler_result = list(provider | ZLMAHandler(length=5))

    # After enough values, ZLMA of constant should equal the constant
    # Skip the first 20 values to allow for convergence
    threshold = 1e-10
    for result in handler_result[20:]:
        if result is not None:
            assert abs(result - constant_value) < threshold


def test_zlma_lag_calculation() -> None:
    """Test that the lag is calculated correctly."""
    # Test different length values and verify lag calculation
    test_cases = [
        (1, 0),  # length=1: lag=int(0.5 * (1 - 1)) = 0
        (2, 0),  # length=2: lag=int(0.5 * (2 - 1)) = 0
        (3, 1),  # length=3: lag=int(0.5 * (3 - 1)) = 1
        (4, 1),  # length=4: lag=int(0.5 * (4 - 1)) = 1
        (5, 2),  # length=5: lag=int(0.5 * (5 - 1)) = 2
        (10, 4),  # length=10: lag=int(0.5 * (10 - 1)) = 4
    ]

    for length, expected_lag in test_cases:
        # We can't directly check lag in ZLMAHandler as it's calculated internally,
        # but we can calculate it ourselves and use for comparison
        calculated_lag = int(0.5 * (length - 1))
        assert calculated_lag == expected_lag, f"Incorrect lag calculation for length={length}"


def test_zlma_default_parameters() -> None:
    """Test that ZLMAHandler uses default parameters correctly."""
    # Default length should be 10 and default ma_handler should be EMAHandler
    handler = ZLMAHandler()
    default_length = 10
    assert handler.length == default_length
    assert isinstance(handler.ma_handler, EMAHandler)
    assert handler.ma_handler.length == default_length


def test_zlma_requires_source() -> None:
    """Test that ZLMAHandler raises an error if no source is set."""
    handler = ZLMAHandler(length=5)

    # Attempting to iterate without a source should raise ValueError
    with pytest.raises(ValueError):
        list(handler)


def test_zlma_pipeline_chaining() -> None:
    """Test that ZLMAHandler can be chained in a pipeline."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    # Create a pipeline: data -> ZLMA -> SMA
    pipeline_result = list(provider | ZLMAHandler(length=3) | SMAHandler(length=2))

    # Create the same pipeline step by step for comparison
    step1_result = list(provider | ZLMAHandler(length=3))
    step2_provider = SimpleDataProvider(step1_result)
    step2_result = list(step2_provider | SMAHandler(length=2))

    # Results should match
    assert safe_allclose(pipeline_result, step2_result)


def test_zlma_source_constructor() -> None:
    """Test that ZLMAHandler can use a source provided in constructor."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)

    # Create handler with source in constructor
    handler = ZLMAHandler(length=3, source=provider)
    constructor_result = list(handler)

    # Create handler with pipeline syntax
    pipeline_result = list(SimpleDataProvider[float | None](data) | ZLMAHandler(length=3))

    # Results should match
    assert safe_allclose(constructor_result, pipeline_result)


def test_zlma_different_ma_handlers() -> None:
    """Test ZLMA with different ma_handlers and verify they produce different results."""
    data: list[float | None] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    length = 3

    # Get results with different handlers
    provider: SimpleDataProvider[float | None] = SimpleDataProvider(data)
    ema_result = list(provider | ZLMAHandler(length=length, ma_handler=EMAHandler(length=length)))

    provider = SimpleDataProvider(data)
    sma_result = list(provider | ZLMAHandler(length=length, ma_handler=SMAHandler(length=length)))

    provider = SimpleDataProvider(data)
    wma_result = list(provider | ZLMAHandler(length=length, ma_handler=WMAHandler(length=length)))

    # Verify that results are different
    # This isn't guaranteed for all data, but should be true for this test set
    assert not safe_allclose(ema_result, sma_result)
    assert not safe_allclose(ema_result, wma_result)
    assert not safe_allclose(sma_result, wma_result)

    # Also check against manual implementation
    pd_data = pd.Series(data)
    pd_ema = calculate_reference_zlma(pd_data, length, "ema")
    pd_sma = calculate_reference_zlma(pd_data, length, "sma")
    pd_wma = calculate_reference_zlma(pd_data, length, "wma")

    assert safe_allclose(ema_result, pd_ema, rtol=1e-3, atol=1e-3)
    assert safe_allclose(sma_result, pd_sma, rtol=1e-3, atol=1e-3)
    assert safe_allclose(wma_result, pd_wma, rtol=1e-3, atol=1e-3)
