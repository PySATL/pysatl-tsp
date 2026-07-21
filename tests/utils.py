from collections import deque
from collections.abc import Callable
from statistics import median

import numpy as np


def safe_allclose(a: list[float | None], b: list[float | None], rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    a_filtered = [x for x in a if x is not None]
    b_filtered = [x for x in b if x is not None]

    if len(a_filtered) != len(b_filtered):
        return False

    if not a_filtered:
        return True

    return bool(np.allclose(np.array(a_filtered), np.array(b_filtered), rtol=rtol, atol=atol))


class _StubOp:
    """Sliding-window operation mirroring one operation of the Julia backend."""

    def __init__(self, length: int, min_periods: int, reduce: Callable[[list[float]], float]) -> None:
        self.length = length
        self.min_periods = min_periods
        self.reduce = reduce
        self.window: deque[float | None] = deque(maxlen=length)

    def step(self, value: float, is_missing: bool) -> tuple[float, bool]:
        self.window.append(None if is_missing else value)
        valid = [v for v in self.window if v is not None]
        if len(valid) < self.min_periods:
            return float("nan"), True
        return self.reduce(valid), False


class _StubChain:
    """Ordered sequence of stub operations, mirroring the Julia ``Chain``."""

    def __init__(self, ops: list[_StubOp]) -> None:
        self.ops = ops

    def step(self, value: float, is_missing: bool) -> tuple[float, bool]:
        for op in self.ops:
            value, is_missing = op.step(value, is_missing)
        return value, is_missing


class JuliaBackendStub:
    """Pure-Python test double for the Julia ``TSPBackend`` module.

    It implements the same surface the handlers rely on, which lets the Python
    side (pipeline integration, fusion, missing-value conversion) be tested
    without a Julia runtime, and doubles as an executable description of the
    contract the Julia implementation must satisfy.

    :ivar chains: Every chain built through this backend, in creation order
    """

    def __init__(self) -> None:
        self.chains: list[_StubChain] = []

    def sma(self, length: int, min_periods: int) -> _StubOp:
        return _StubOp(length, min_periods, lambda values: sum(values) / len(values))

    def median_filter(self, length: int, min_periods: int) -> _StubOp:
        return _StubOp(length, min_periods, median)

    def chain_of(self, *ops: _StubOp) -> _StubChain:
        chain = _StubChain(list(ops))
        self.chains.append(chain)
        return chain

    def step_chain(self, chain: _StubChain, value: float, is_missing: bool) -> tuple[float, bool]:
        return chain.step(value, is_missing)
