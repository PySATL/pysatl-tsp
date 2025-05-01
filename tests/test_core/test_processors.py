from typing import Any, Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from pysatl_tsp.core.data_providers import DataProvider, SimpleDataProvider
from pysatl_tsp.core.processor import (
    MappingHandler,
    OfflineFilterHandler,
    OfflineSamplingHandler,
    OnlineFilterHandler,
    OnlineSamplingHandler,
)
from pysatl_tsp.core.scrubber import ScrubberWindow


class TestMappingHandlerWithHypothesis:
    @st.composite
    def data_provider_strategy(draw: DrawFn) -> tuple[DataProvider[Any], Callable[[Any], Any]]:
        type_choices = st.sampled_from(
            [
                (st.integers(), lambda x: x * 2),
                (st.text(), lambda s: s.upper()),
                (st.lists(st.integers()), lambda lst: sum(lst)),
                (st.dictionaries(st.text(), st.integers()), lambda d: len(d)),
            ]
        )

        data_type, func = draw(type_choices)
        data = draw(st.lists(data_type, max_size=20))
        return SimpleDataProvider(data), func

    @given(data_provider_strategy())
    def test_general_mapping(self, provider_and_func: tuple[DataProvider[Any], Callable[[Any], Any]]) -> None:
        provider, map_func = provider_and_func
        mapper = MappingHandler(map_func=map_func, source=provider)

        input_data = list(provider)
        try:
            expected = [map_func(x) for x in input_data]
            result = list(mapper)
            assert result == expected
        except Exception as e:
            with pytest.raises(type(e)):
                list(mapper)

    @st.composite
    def single_arg_functions(draw: DrawFn) -> Callable[[Any], Any]:
        return draw(st.functions(like=lambda x: x, returns=st.integers(), pure=True))

    @given(st.lists(st.integers(min_value=0), min_size=10, max_size=20), single_arg_functions())
    def test_lazy_evaluation(self, input_data: list[int], map_func: Callable[[int], Any]) -> None:
        provider = SimpleDataProvider(input_data)
        call_counter = 0

        def wrapped_map(x: int) -> Any:
            nonlocal call_counter
            call_counter += 1
            return map_func(x)

        mapper = MappingHandler(wrapped_map, provider)
        iterator = iter(mapper)

        consumed = min(5, len(input_data))
        for _ in range(consumed):
            next(iterator)

        assert call_counter == consumed

        for _ in iterator:
            pass

        assert call_counter == len(input_data)

    @given(st.lists(st.integers()))
    def test_chain_of_handlers(self, input_data: list[int]) -> None:
        provider = SimpleDataProvider(input_data)

        mapper1 = MappingHandler(lambda x: x * 2, provider)
        mapper2 = MappingHandler(lambda x: x + 1, mapper1)

        expected = [(x * 2) + 1 for x in input_data]
        assert list(mapper2) == expected

    @given(st.lists(st.integers(min_value=1)))
    def test_error_handling_with_hypothesis(self, input_data: list[int]) -> None:
        provider = SimpleDataProvider(input_data)

        def safe_division(x: int) -> float:
            return x / (x % 2)

        mapper = MappingHandler(safe_division, provider)

        try:
            list(mapper)
        except ZeroDivisionError:
            assert any(x % 2 == 0 for x in input_data)

    @given(st.lists(st.integers(), min_size=1))
    def test_ordering_preservation(self, input_data: list[int]) -> None:
        provider = SimpleDataProvider(input_data)
        mapper = MappingHandler(lambda x: x, provider)
        assert list(mapper) == input_data

    @given(st.lists(st.one_of(st.none(), st.integers())))
    def test_none_handling(self, input_data: list[int]) -> None:
        provider = SimpleDataProvider(input_data)
        mapper = MappingHandler(lambda x: x is None, provider)
        expected = [x is None for x in input_data]
        assert list(mapper) == expected

    @given(st.lists(st.tuples(st.integers(), st.text())))
    def test_complex_structures(self, input_data: list[tuple[int, str]]) -> None:
        provider = SimpleDataProvider(input_data)

        def transform(item: tuple[int, str]) -> str:
            num, text = item
            return f"{text}_{num}"

        mapper = MappingHandler(transform, provider)
        expected = [f"{text}_{num}" for num, text in input_data]
        assert list(mapper) == expected

    def test_empty_source_handling(self) -> None:
        provider: SimpleDataProvider[Any] = SimpleDataProvider([])
        mapper = MappingHandler(lambda x: x, provider)
        assert list(mapper) == []

    def test_sampling(self) -> None:
        res = SimpleDataProvider([1, 2, 3, 4, 5]) | OfflineSamplingHandler(
            sampling_rule=lambda s: list(range(1, len(s), 2))
        )
        assert list(res) == [1, 3, 5]


class TestOnlineFilterHandler:
    def test_initialization(self) -> None:
        def simple_filter(history: ScrubberWindow[float], config: int) -> float:
            return sum(history) / len(history)

        config = 5
        handler: OnlineFilterHandler[float, float] = OnlineFilterHandler(
            filter_func=simple_filter, filter_config=config
        )
        assert handler.filter_func is simple_filter
        assert handler.filter_config == config
        assert handler.source is None

    def test_moving_average_filter(self) -> None:
        data: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        provider: SimpleDataProvider[float] = SimpleDataProvider(data)

        def moving_average(history: ScrubberWindow[float], window_size: int) -> float:
            if len(history) < window_size:
                return sum(history) / len(history)
            return sum(history[len(history) - window_size :]) / window_size

        handler: OnlineFilterHandler[float, float] = OnlineFilterHandler(
            filter_func=moving_average, filter_config=3, source=provider
        )
        result: list[float] = list(handler)

        expected: list[float] = [1.0, 1.5, 2.0, 3.0, 4.0]
        assert result == expected

    def test_exponential_smoothing(self) -> None:
        data: list[float] = [1.0, 3.0, 5.0, 2.0, 4.0]
        provider: SimpleDataProvider[float] = SimpleDataProvider(data)

        def exp_smooth(history: ScrubberWindow[float], alpha: float) -> float:
            if len(history) == 1:
                return history[0]
            prev: float = exp_smooth(history[: len(history) - 1], alpha)
            return alpha * history[-1] + (1 - alpha) * prev

        handler: OnlineFilterHandler[float, float] = OnlineFilterHandler(
            filter_func=exp_smooth, filter_config=0.3, source=provider
        )
        result: list[float] = list(handler)

        expected: list[float] = [
            1.0,
            1.0 + 0.3 * (3.0 - 1.0),
            1.6 + 0.3 * (5.0 - 1.6),
            2.62 + 0.3 * (2.0 - 2.62),
            2.434 + 0.3 * (4.0 - 2.434),
        ]
        np.testing.assert_almost_equal(result, expected, decimal=4)

    def test_empty_series(self) -> None:
        provider: SimpleDataProvider[float] = SimpleDataProvider([])

        def empty_filter(h: ScrubberWindow[float], c: None) -> float:
            return sum(h) / len(h) if h else 0

        handler: OnlineFilterHandler[float, float] = OnlineFilterHandler(
            filter_func=empty_filter, filter_config=None, source=provider
        )
        result: list[float] = list(handler)
        assert result == []

    def test_source_not_set(self) -> None:
        def sum_filter(h: ScrubberWindow[int], c: None) -> int:
            return sum(h)

        handler: OnlineFilterHandler[int, int] = OnlineFilterHandler(filter_func=sum_filter)
        with pytest.raises(ValueError, match="Source is not set"):
            list(handler)


class TestOfflineFilterHandler:
    def test_initialization(self) -> None:
        def simple_filter(series: ScrubberWindow[int], config: int) -> list[int]:
            return [x * config for x in series]

        config = 2
        handler: OfflineFilterHandler[int, int] = OfflineFilterHandler(filter_func=simple_filter, filter_config=config)
        assert handler.filter_func is simple_filter
        assert handler.filter_config == config
        assert handler.source is None

    def test_savgol_filter_simulation(self) -> None:
        data: list[float] = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        provider: SimpleDataProvider[float] = SimpleDataProvider(data)

        def smooth(series: ScrubberWindow[float], window_size: int) -> list[float]:
            result: list[float] = []
            for i in range(len(series)):
                start: int = max(0, i - window_size // 2)
                end: int = min(len(series), i + window_size // 2 + 1)
                result.append(sum(series[start:end]) / (end - start))
            return result

        handler: OfflineFilterHandler[float, float] = OfflineFilterHandler(
            filter_func=smooth, filter_config=3, source=provider
        )
        result: list[float] = list(handler)

        expected: list[float] = [1.5, 2.0, 2.33333, 2.0, 1.66667, 2.0, 2.5]
        np.testing.assert_almost_equal(result, expected, decimal=4)

    def test_empty_series(self) -> None:
        provider: SimpleDataProvider[float] = SimpleDataProvider([])

        def identity_filter(s: ScrubberWindow[float], c: None) -> list[float]:
            return list(s.values)

        handler: OfflineFilterHandler[float, float] = OfflineFilterHandler(
            filter_func=identity_filter, filter_config=None, source=provider
        )
        result: list[float] = list(handler)
        assert result == []


class TestOnlineSamplingHandler:
    def test_initialization(self) -> None:
        def simple_rule(window: ScrubberWindow[int]) -> bool:
            three = 3
            return len(window) >= three

        handler: OnlineSamplingHandler[int] = OnlineSamplingHandler(sampling_rule=simple_rule)
        assert handler.sampling_rule is simple_rule
        assert handler.source is None

    def test_sampling_every_n(self) -> None:
        data: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        provider: SimpleDataProvider[int] = SimpleDataProvider(data)

        def segment_every_three(window: ScrubberWindow[int]) -> bool:
            three = 3
            return len(window) >= three

        handler: OnlineSamplingHandler[int] = OnlineSamplingHandler(sampling_rule=segment_every_three, source=provider)
        result: list[int] = list(handler)

        expected: list[int] = [3, 6, 9]
        assert result == expected

    def test_sampling_on_threshold(self) -> None:
        data: list[int] = [1, 2, 10, 3, 4, 15, 5, 6]
        provider: SimpleDataProvider[int] = SimpleDataProvider(data)

        def threshold_rule(window: ScrubberWindow[int]) -> bool:
            threshold = 8
            if not window:
                return False
            return window[-1] > threshold

        handler: OnlineSamplingHandler[int] = OnlineSamplingHandler(sampling_rule=threshold_rule, source=provider)
        result: list[int] = list(handler)

        expected: list[int] = [10, 15, 6]
        assert result == expected

    def test_empty_series(self) -> None:
        provider: SimpleDataProvider[int] = SimpleDataProvider([])

        def always_true(w: ScrubberWindow[int]) -> bool:
            return True

        handler: OnlineSamplingHandler[int] = OnlineSamplingHandler(sampling_rule=always_true, source=provider)
        result: list[int] = list(handler)
        assert result == []

    def test_source_not_set(self) -> None:
        def always_true(w: ScrubberWindow[int]) -> bool:
            return True

        handler: OnlineSamplingHandler[int] = OnlineSamplingHandler(sampling_rule=always_true)
        with pytest.raises(ValueError, match="Source is not set"):
            list(handler)


class TestOfflineSamplingHandler:
    def test_initialization(self) -> None:
        def simple_rule(window: ScrubberWindow[int]) -> list[int]:
            return [3, 6]

        handler: OfflineSamplingHandler[int] = OfflineSamplingHandler(sampling_rule=simple_rule)
        assert handler.sampling_rule is simple_rule
        assert handler.source is None

    def test_sampling_at_indices(self) -> None:
        data: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        provider: SimpleDataProvider[int] = SimpleDataProvider(data)

        def segment_at_indices(window: ScrubberWindow[int]) -> list[int]:
            return [3, 6]

        handler: OfflineSamplingHandler[int] = OfflineSamplingHandler(sampling_rule=segment_at_indices, source=provider)
        result: list[int] = list(handler)

        expected: list[int] = [3, 6, 9]
        assert result == expected

    def test_sampling_with_custom_rule(self) -> None:
        data: list[int] = [1, 3, 7, 9, 5, 2, 8, 10]
        provider: SimpleDataProvider[int] = SimpleDataProvider(data)

        def segment_on_big_difference(window: ScrubberWindow[int]) -> list[int]:
            big_difference = 3
            indices: list[int] = []
            for i in range(1, len(window)):
                if abs(window[i] - window[i - 1]) > big_difference:
                    indices.append(i)
            return indices

        handler: OfflineSamplingHandler[int] = OfflineSamplingHandler(
            sampling_rule=segment_on_big_difference, source=provider
        )
        result: list[int] = list(handler)

        expected: list[int] = [3, 9, 2, 10]
        assert result == expected

    def test_source_not_set(self) -> None:
        def empty_rule(w: ScrubberWindow[int]) -> list[int]:
            return []

        handler: OfflineSamplingHandler[int] = OfflineSamplingHandler(sampling_rule=empty_rule)
        with pytest.raises(ValueError, match="Source is not set"):
            list(handler)


class TestIntegration:
    def test_online_filter_with_sampling(self) -> None:
        data: list[float] = [1.0, 5.0, 10.0, 8.0, 3.0, 7.0, 15.0, 13.0, 9.0]
        provider: SimpleDataProvider[float] = SimpleDataProvider(data)

        def exp_smooth(history: ScrubberWindow[float], alpha: float) -> float:
            if len(history) == 1:
                return history[0]
            prev: float = exp_smooth(history[: len(history) - 1], alpha)
            return alpha * history[-1] + (1 - alpha) * prev

        filter_handler: OnlineFilterHandler[float, float] = OnlineFilterHandler(
            filter_func=exp_smooth, filter_config=0.3, source=provider
        )

        threshold = 8.0

        def threshold_rule(window: ScrubberWindow[float]) -> bool:
            if not window:
                return False
            return window[-1] > threshold

        sampling_handler: OnlineSamplingHandler[float] = OnlineSamplingHandler(
            sampling_rule=threshold_rule, source=filter_handler
        )

        result: list[float] = list(sampling_handler)

        assert len(result) > 0

        for value in result:
            assert value > threshold

    def test_offline_filter_chain(self) -> None:
        """Проверяет цепочку из нескольких оффлайн-фильтров."""
        data: list[int] = [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
        provider: SimpleDataProvider[int] = SimpleDataProvider(data)

        def smooth(series: ScrubberWindow[int], window_size: int) -> list[float]:
            result: list[float] = []
            for i in range(len(series)):
                start: int = max(0, i - window_size // 2)
                end: int = min(len(series), i + window_size // 2 + 1)
                result.append(sum(series[start:end]) / (end - start))
            return result

        smooth_filter: OfflineFilterHandler[int, float] = OfflineFilterHandler(
            filter_func=smooth, filter_config=3, source=provider
        )

        def standardize(series: ScrubberWindow[float], _: None) -> list[float]:
            mean: float = sum(series) / len(series)
            variance: float = sum((x - mean) ** 2 for x in series) / len(series)
            std: float = variance**0.5
            return [(x - mean) / std for x in series]

        standardize_filter: OfflineFilterHandler[float, float] = OfflineFilterHandler(
            filter_func=standardize, filter_config=None, source=smooth_filter
        )

        result: list[float] = list(standardize_filter)

        mean: float = sum(result) / len(result)
        variance: float = sum((x - mean) ** 2 for x in result) / len(result)
        std: float = variance**0.5
        assert np.isclose(abs(mean), 0)
        assert np.isclose(std - 1.0, 0)
