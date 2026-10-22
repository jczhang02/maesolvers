"""Presets: named solvers for the common pairs of a scheme and an iteration.

A preset (ADR 0004) is an iteration whose scheme is built from flat
parameters, so the simple call ``FDPoisson(n=64).solve(problem)`` stays and
its :attr:`~maesolvers.core.solver.Solver.hparams` stay flat. The presets are
what :data:`maesolvers.registry.SOLVERS` and the comparison tables list, and
like every solver they declare their support on the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from maesolvers.classical.iterations import Newton, PoissonIteration
from maesolvers.classical.schemes import Standard9


@dataclass(frozen=True, kw_only=True)
class _Standard9Preset:
    """The fields and support of a preset on :class:`Standard9`.

    Listed first among a preset's bases, so its ``__post_init__`` builds the
    scheme from ``n`` and ``spline_degree`` before the iteration checks it.
    """

    domains: ClassVar[tuple[type, ...]] = Standard9.domains
    boundaries: ClassVar[tuple[type, ...]] = Standard9.boundaries
    dims: ClassVar[tuple[int, ...]] = Standard9.dims

    scheme: Standard9 = field(init=False, repr=False)
    n: int = Standard9.n
    spline_degree: int = Standard9.spline_degree

    def __post_init__(self) -> None:
        scheme = Standard9(n=self.n, spline_degree=self.spline_degree)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", scheme.n)
        object.__setattr__(self, "spline_degree", scheme.spline_degree)
        super().__post_init__()  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class FDNewton(_Standard9Preset, Newton):
    """The standard 9-point scheme solved by damped Newton.

    ``FDNewton(n=n, spline_degree=k, ...)`` is
    ``Newton(scheme=Standard9(n=n, spline_degree=k), ...)``. On a singular
    problem it may converge to a non-convex discrete root
    (``benchmarks/standard_fd.md``, Finding 2).

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5 and at most ``n``.
    """


@dataclass(frozen=True, kw_only=True)
class FDPoisson(_Standard9Preset, PoissonIteration):
    """The standard 9-point scheme solved by the Poisson iteration.

    ``FDPoisson(n=n, spline_degree=k, ...)`` is
    ``PoissonIteration(scheme=Standard9(n=n, spline_degree=k), ...)``, the
    "standard finite differences" column of Froese-Oberman 2011, Table 3.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001); at least 2.
    max_iter
        Most iterations.
    tol
        Stop when successive iterates differ by at most ``tol`` at every node.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5 and at most ``n``.
    """
