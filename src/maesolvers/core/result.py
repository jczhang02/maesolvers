"""What a solver returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from maesolvers.core.domains import Field, HessianField
from maesolvers.core.problem import Problem


@dataclass(frozen=True, eq=False)
class Result:
    """The solution of a :class:`~maesolvers.core.problem.Problem` by one method.

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
        NaN where it defines none (boundary nodes), or None.
    info
        Run record: ``time`` (seconds), ``iterations``, ``converged``,
        ``unknowns`` and method-specific entries.
    hparams
        The solver's configuration, ``{"class_path": ..., "init_args": ...}``,
        enough to build the same solver again.
    """

    problem: Problem
    u: Field
    hessian: HessianField | None = None
    nodes: np.ndarray | None = None
    values: np.ndarray | None = None
    node_hessian: np.ndarray | None = None
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
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        object.__setattr__(self, "hparams", MappingProxyType(dict(self.hparams)))

    def _check_node_arrays(self) -> None:
        d = self.problem.dim
        nodes = np.asarray(self.nodes, dtype=float)
        values = np.asarray(self.values, dtype=float)
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
            hess = np.asarray(self.node_hessian, dtype=float)
            if hess.shape != (m, d, d):
                raise ValueError(
                    f"Result.node_hessian must have shape ({m}, {d}, {d}), "
                    f"got {hess.shape}"
                )
            object.__setattr__(self, "node_hessian", hess)
