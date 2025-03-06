import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from claspy.segmentation import BinaryClaSPSegmentation
from claspy.data_loader import load_tssb_dataset

from ts_flow.core.scrubber import ScrubberWindow, LinearScrubber, SegmentationScrubber, OnlineStrategy, OfflineStrategy
from ts_flow.core.data_providers import SimpleDataProvider


numeric_lists = st.lists(st.floats(allow_nan=False, allow_infinity=False))


class TestLinearScrubber:
    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(1, 100), st.floats(0.01, 1))
    def test_get_windows(self, data_length, window_length, shift_factor):
        data = [float(i) for i in range(data_length)]
        scrubber = LinearScrubber(window_length, shift_factor, SimpleDataProvider(data))
        cur_index = 0
        for window in iter(scrubber):
            assert len(window.values) == len(window.indices)
            assert np.array_equal(window.values, np.fromiter(data[cur_index : cur_index + window_length], np.float64))
            cur_index += max(1, int(window_length * shift_factor))


class TestOfflineSegmentationScrubber:
    @given(numeric_lists)
    def test_fixed_segments(self, data: list[float]):
        def segment_rule(x: list[float]) -> list[int]:
            return list(range(3, len(x), 3))

        provider = SimpleDataProvider(data)
        scrubber = SegmentationScrubber(strategy=OfflineStrategy(segment_rule), source=provider)

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
            assert window.values == expected_segments[i]
            assert window.indices == indices[i]

    def test_empty_source(self):
        scrubber = SegmentationScrubber(strategy=OfflineStrategy(lambda x: []))
        scrubber.source = SimpleDataProvider([])

        output = list(iter(scrubber))
        assert len(output) == 1
        assert output[0].values == []

    def test_claspy_integration_consistency(self):
        _, _, _, ts_np = load_tssb_dataset(names=("CricketX",)).iloc[0, :]

        clasp = BinaryClaSPSegmentation()
        expected_cps = clasp.fit_predict(ts_np)

        provider = SimpleDataProvider(ts_np.tolist())
        scrubber = SegmentationScrubber(
            strategy=OfflineStrategy(lambda data: clasp.fit_predict(np.array(data)).tolist()), source=provider
        )

        segments = list(scrubber)
        result_cps = [seg.indices[-1] for seg in segments[:-1]]

        np.testing.assert_array_equal(
            np.array(result_cps) + 1,
            expected_cps,
            err_msg="CLAASP results differ between direct call and scrubber integration",
        )

        reconstructed = np.concatenate([seg.values for seg in segments])
        np.testing.assert_allclose(reconstructed, ts_np, err_msg="Data reconstruction mismatch")


class TestOnlineSegmentationScrubber:
    @given(numeric_lists)
    def test_fixed_window_size(self, data: list[float]):
        def online_rule(buffer: list[float]) -> bool:
            return len(buffer) >= 5

        provider = SimpleDataProvider(data)
        scrubber = SegmentationScrubber(
            OnlineStrategy(segmentation_rule=online_rule, max_segment_size=10), source=provider
        )

        expected = []
        current_segment = []
        for i, item in enumerate(data):
            current_segment.append(item)
            if len(current_segment) >= 5:
                expected.append(
                    ScrubberWindow(
                        values=current_segment.copy(), indices=list(range(i - len(current_segment) + 1, i + 1))
                    )
                )
                current_segment = []
        if current_segment:
            expected.append(
                ScrubberWindow(
                    values=current_segment.copy(), indices=list(range(len(data) - len(current_segment), len(data)))
                )
            )

        result = list(scrubber)
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res.values == exp.values
            assert res.indices == exp.indices

    def test_buffer_overflow(self):
        data = list(range(20))

        def rule(buffer: list[int]) -> bool:
            return len(buffer) >= 3

        scrubber = SegmentationScrubber(
            OnlineStrategy(segmentation_rule=rule, max_segment_size=5), source=SimpleDataProvider(data)
        )

        output = list(scrubber)
        expected = []
        for i in range(0, 20, 3):
            segment = data[i : i + 3]
            expected.append(ScrubberWindow(segment, segment))
        assert len(output) == 7
        assert output == expected

    def test_no_segments(self):
        scrubber = SegmentationScrubber(strategy=OnlineStrategy(lambda b: False))
        scrubber.source = SimpleDataProvider([1, 2, 3])
        assert list(scrubber) == [ScrubberWindow([1, 2, 3], [0, 1, 2])]


def test_scrubber_window_slicing():
    window = ScrubberWindow(values=[0.1, 0.2, 0.3, 0.4], indices=[10, 11, 12, 13])

    sliced = window[1:3]
    assert sliced.values == [0.2, 0.3]
    assert sliced.indices == [11, 12]

    assert window[2] == 0.3
