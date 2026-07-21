"""Common base for Julia-backed handlers, including cross-handler fusion.

A Julia-backed handler does not describe *how* to compute its result in Python;
it only describes *which* Julia operation it stands for. That indirection makes
it possible to detect a run of adjacent Julia-backed handlers in a pipeline and
evaluate it as a single Julia chain, so that a pipeline such as

    provider | JuliaSMAHandler(...) | JuliaMedianFilterHandler(...)

crosses the language boundary once per observation instead of once per handler.

Fusion is an implementation detail of this subpackage and changes nothing in
``pysatl_tsp.core``: the ``Handler`` contract, the ``|`` operator and the
``Pipeline`` class are used exactly as they are. Handlers are only inspected,
never rewired, so a fused handler remains usable on its own and the "a handler
has a single source" rule still holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pysatl_tsp.core import Handler
from pysatl_tsp.core.handler import Pipeline

from ._bridge import get_backend

__all__ = ["JuliaHandler", "JuliaOpSpec"]

#: Value sent across the boundary for a missing observation. The accompanying
#: ``is_missing`` flag is what the backend actually acts on, so this is only a
#: placeholder that keeps the wire type a plain float.
_MISSING_INPUT = float("nan")


@dataclass(frozen=True)
class JuliaOpSpec:
    """Declarative description of a single Julia operation.

    The spec names a factory function exported by the Julia backend together
    with its positional arguments. It carries no state, so the same handler can
    be described repeatedly and each description yields a fresh Julia operation.

    :param factory: Name of the factory function in the Julia backend module
    :param args: Positional arguments passed to that factory
    """

    factory: str
    args: tuple[Any, ...] = field(default_factory=tuple)


def _producing_handler(node: Handler[Any, Any] | None) -> Handler[Any, Any] | None:
    """Resolve which handler actually produces the items of ``node``.

    ``Pipeline`` delegates iteration to its second handler, so a pipeline is
    transparent for the purposes of fusion and is unwrapped here.

    :param node: A handler, a pipeline, or None
    :return: The handler that yields the data, or None if there is no source
    """
    while isinstance(node, Pipeline):
        node = node.second
    return node


class JuliaHandler(Handler[float | None, float | None], ABC):
    """Base class for handlers whose computation is delegated to Julia.

    Subclasses only implement :meth:`_julia_op_spec`; iteration, fusion with
    adjacent Julia handlers and conversion of missing values are handled here.
    A handler that is not preceded by another Julia handler simply runs as a
    chain of length one, so fused and unfused execution follow the same code
    path and cannot drift apart semantically.

    :param source: The handler providing input data, defaults to None
    """

    @abstractmethod
    def _julia_op_spec(self) -> JuliaOpSpec:
        """Describe this handler as a Julia operation.

        :return: Specification of the Julia operation this handler stands for
        """

    def _fuse(self) -> tuple[Handler[Any, float | None], list[JuliaOpSpec]]:
        """Collect the longest run of Julia handlers ending at this one.

        The source chain is walked upstream while it keeps yielding Julia
        handlers. Walking stops at the first handler that is not Julia-backed,
        which becomes the data source for the fused chain.

        :return: The upstream source and the operation specs in upstream-to-downstream order
        :raises ValueError: If no source has been set
        """
        specs = [self._julia_op_spec()]

        node: Handler[Any, Any] | None = self.source
        while isinstance(producer := _producing_handler(node), JuliaHandler):
            specs.append(producer._julia_op_spec())
            node = producer.source

        if node is None:
            raise ValueError("Source is not set")

        specs.reverse()
        return node, specs

    def __iter__(self) -> Iterator[float | None]:
        """Stream observations through the fused Julia chain.

        :return: Iterator yielding the results of the whole fused chain
        :raises ValueError: If no source has been set
        """
        source, specs = self._fuse()

        backend = get_backend()
        chain = backend.chain_of(*(getattr(backend, spec.factory)(*spec.args) for spec in specs))
        step_chain = backend.step_chain

        for value in source:
            if value is None:
                result, missing = step_chain(chain, _MISSING_INPUT, True)
            else:
                result, missing = step_chain(chain, float(value), False)
            yield None if bool(missing) else float(result)
