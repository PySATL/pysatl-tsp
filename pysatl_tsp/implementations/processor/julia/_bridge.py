"""Bridge between Python handlers and the Julia backend.

The Julia runtime is heavy to start, so it is initialized lazily and exactly
once per process. The `juliacall` package is an optional dependency: importing
this module is always safe, an informative error is raised only when a Julia
handler is actually used without `juliacall` installed.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["get_backend"]

_JULIA_SOURCE = Path(__file__).with_name("backend.jl")


@lru_cache(maxsize=1)
def get_backend() -> Any:
    """Load the Julia ``TSPBackend`` module and return a handle to it.

    The Julia source file is included into the Julia ``Main`` namespace on the
    first call; subsequent calls return the cached module handle.

    :return: Handle to the Julia ``TSPBackend`` module
    :raises ImportError: If the optional dependency ``juliacall`` is not installed
    """
    try:
        juliacall = importlib.import_module("juliacall")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Julia-backed handlers require the optional dependency 'juliacall'. "
            "Install it with: poetry install --with julia"
        ) from exc

    jl = juliacall.Main
    jl.seval(f'include(raw"{_JULIA_SOURCE}")')
    return jl.TSPBackend
