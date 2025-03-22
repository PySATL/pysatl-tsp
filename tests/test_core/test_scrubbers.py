from collections import deque
from typing import Any

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from ts_flow.core.data_providers import SimpleDataProvider
from ts_flow.core.scrubber import (
    LinearScrubber,
    OfflineSegmentationScrubber,
    OnlineSegmentationScrubber,
    ScrubberWindow,
)

numeric_lists = st.lists(st.floats(allow_nan=False, allow_infinity=False))


class TestLinearScrubber:
    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(1, 100), st.floats(0.01, 1))
    def test_get_windows(self, data_length: int, window_length: int, shift_factor: float) -> None:
        data = [float(i) for i in range(data_length)]
        scrubber = LinearScrubber(window_length, shift_factor, SimpleDataProvider(data))
        cur_index = 0
        for window in iter(scrubber):
            assert len(window.values) == len(window.indices)
            assert np.array_equal(window.values, np.fromiter(data[cur_index : cur_index + window_length], np.float64))
            cur_index += max(1, int(window_length * shift_factor))


class TestOfflineSegmentationScrubber:
    @given(numeric_lists)
    def test_fixed_segments(self, data: list[float]) -> None:
        def segment_rule(x: ScrubberWindow[float]) -> list[int]:
            return list(range(3, len(x), 3))

        provider = SimpleDataProvider(data)
        scrubber = OfflineSegmentationScrubber(segmentation_rule=segment_rule, source=provider)

        expected_segments = []
        indices = []
        for i in range(0, len(data), 3):
            segment = data[i : i + 3]
            if len(segment) > 0:
                expected_segments.append(segment)
                indices.append(list(range(i, min(i + 3, len(data)))))

        if not expected_segments:
            expected_segments.append([])
            indices.append([])

        for i, window in enumerate(scrubber):
            assert window.values == deque(expected_segments[i])
            assert window.indices == deque(indices[i])

    def test_empty_source(self) -> None:
        scrubber: OfflineSegmentationScrubber[Any] = OfflineSegmentationScrubber(segmentation_rule=lambda _: [])
        scrubber.source = SimpleDataProvider([])

        output = list(iter(scrubber))
        assert len(output) == 1
        assert output[0].values == deque([])


class TestOnlineSegmentationScrubber:
    @given(numeric_lists)
    def test_fixed_window_size(self, data: list[float]) -> None:
        rule_threshold = 5

        def online_rule(buffer: ScrubberWindow[float]) -> bool:
            return len(buffer) >= rule_threshold

        provider = SimpleDataProvider(data)
        scrubber = OnlineSegmentationScrubber(segmentation_rule=online_rule, max_segment_size=10, source=provider)

        expected: list[ScrubberWindow[float]] = []
        current_segment: ScrubberWindow[float] = ScrubberWindow(deque())
        for i, item in enumerate(data):
            current_segment.append(item, i)
            if len(current_segment) >= rule_threshold:
                expected.append(current_segment.copy())
                current_segment.clear()

        if current_segment:
            expected.append(current_segment.copy())

        result = list(scrubber)
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res.values == exp.values
            assert res.indices == exp.indices

    def test_buffer_overflow(self) -> None:
        data = list(range(20))
        rule_threshold = 3

        def rule(buffer: ScrubberWindow[int]) -> bool:
            return len(buffer) >= rule_threshold

        scrubber = OnlineSegmentationScrubber(
            segmentation_rule=rule, max_segment_size=5, source=SimpleDataProvider(data)
        )

        output = list(scrubber)
        expected: list[ScrubberWindow[int]] = []
        for i in range(0, 20, 3):
            segment = data[i : i + 3]
            expected.append(ScrubberWindow(deque(segment), deque(segment)))
        expected_size = len(expected)
        assert len(output) == expected_size
        assert output == expected

    def test_no_segments(self) -> None:
        scrubber: OnlineSegmentationScrubber[int] = OnlineSegmentationScrubber(segmentation_rule=lambda _: False)
        source_data = [1, 2, 3]
        scrubber.source = SimpleDataProvider(source_data)
        assert list(scrubber) == [ScrubberWindow(deque(source_data))]


def test_scrubber_window_slicing() -> None:
    window = ScrubberWindow(values=deque([0.1, 0.2, 0.3, 0.4]), indices=deque([10, 11, 12, 13]))

    sliced = window[1:3]
    assert sliced.values == deque([0.2, 0.3])
    assert sliced.indices == deque([11, 12])

    assert window[2] == window.values[2]


def test_pipe() -> None:
    data = list(range(20))
    scrubber1 = SimpleDataProvider(data) | LinearScrubber(window_length=3) | LinearScrubber(window_length=2)
    scrubber2 = LinearScrubber(window_length=2, source=LinearScrubber(window_length=3, source=SimpleDataProvider(data)))
    assert list(scrubber1) == list(scrubber2)
