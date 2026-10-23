"""What a solver returns."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import numpy as np

from maesolvers.core.domains import Field, HessianField
from maesolvers.core.problem import Problem


@dataclass(frozen=True, eq=False)
class Result:
    """The solution of a :class:`~maesolvers.core.problem.Problem` by one method.

    Every field after ``u`` is keyword-only. The node arrays are stored as
    read-only copies (float, boolean for ``interior``), ``info`` and
    ``hparams`` as deep copies in plain
    dicts, so a result does not change when the caller's data does and its
    records serialise with ``json`` or YAML as they are.

    Parameters
    ----------
    problem
        The problem that was solved.
    u
        The solution anywhere in the domain: an interpolant for a grid method,
        the network for a neural method. A :data:`Field`.
    hessian
        The Hessian of ``u`` anywhere in the domain (autograd for a neural
        method), or None.
    nodes
        Shape ``(M, d)``: the points where a grid method itself defines the
        solution, or None.
    values
        Shape ``(M,)``: the solution at ``nodes``; given exactly when ``nodes``
        is.
    node_hessian
        Shape ``(M, d, d)``: the method's own discrete Hessian at ``nodes``,
        or None. :func:`~maesolvers.metrics.evaluate` reads it at the interior
        nodes only and ignores the entries at boundary nodes, which may be NaN.
    interior
        Shape ``(M,)``, boolean: which of ``nodes`` are the method's interior
        nodes, where :func:`~maesolvers.metrics.evaluate` reads
        ``node_hessian``; or None. Set it whenever the method knows it: when
        it is None, ``evaluate`` asks ``problem.domain.interior(nodes)``, a
        floating-point test that can count nodes placed on a curved boundary
        as interior.
    info
        Run record: ``time`` (seconds), ``iterations``, ``converged``,
        ``history`` (the convergence measure after every step), ``unknowns``
        and method-specific entries. ``stop``, when present, says why the run
        ended (such as ``"converged"`` or ``"max_steps"``), with one value
        that every solver shares: ``"diverged"`` marks a failed run, which
        :func:`~maesolvers.metrics.summarize` counts as a failed seed whatever
        its error.
    hparams
        The solver's configuration, ``{"class_path": ..., "init_args": ...}``,
        enough to build the same solver again.
    """

    problem: Problem
    u: Field
    _: KW_ONLY
    hessian: HessianField | None = None
    nodes: np.ndarray | None = None
    values: np.ndarray | None = None
    node_hessian: np.ndarray | None = None
    interior: np.ndarray | None = None
    info: Mapping[str, Any] = field(default_factory=dict)
    hparams: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.u):
            raise TypeError("Result.u must be callable")
        if self.hessian is not None and not callable(self.hessian):
            raise TypeError("Result.hessian must be callable or None")
        if (self.nodes is None) != (self.values is None):
            raise ValueError("Result.nodes and Result.values go together")
        if self.nodes is not None:
            self._check_node_arrays()
        elif self.node_hessian is not None:
            raise ValueError("Result.node_hessian needs Result.nodes")
        elif self.interior is not None:
            raise ValueError("Result.interior needs Result.nodes")
        object.__setattr__(self, "info", copy.deepcopy(dict(self.info)))
        object.__setattr__(self, "hparams", copy.deepcopy(dict(self.hparams)))

    def _check_node_arrays(self) -> None:
        d = self.problem.dim
        nodes = _read_only(self.nodes)
        values = _read_only(self.values)
        if nodes.ndim != 2 or nodes.shape[1] != d:
            raise ValueError(
                f"Result.nodes must have shape (M, {d}), got {nodes.shape}"
            )
        m = nodes.shape[0]
        if values.shape != (m,):
            raise ValueError(
                f"Result.values must have shape ({m},), got {values.shape}"
            )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "values", values)
        if self.node_hessian is not None:
            hess = _read_only(self.node_hessian)
            if hess.shape != (m, d, d):
                raise ValueError(
                    f"Result.node_hessian must have shape ({m}, {d}, {d}), "
                    f"got {hess.shape}"
                )
            object.__setattr__(self, "node_hessian", hess)
        if self.interior is not None:
            interior = np.array(self.interior)
            if interior.dtype != bool:
                raise TypeError(
                    "Result.interior must be a boolean mask, got dtype "
                    f"{interior.dtype}"
                )
            if interior.shape != (m,):
                raise ValueError(
                    f"Result.interior must have shape ({m},), got {interior.shape}"
                )
            interior.setflags(write=False)
            object.__setattr__(self, "interior", interior)


def _read_only(array: object) -> np.ndarray:
    """Return a read-only float copy of ``array``."""
    out = np.array(array, dtype=float)
    out.setflags(write=False)
    return out
