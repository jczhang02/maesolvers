"""Presets: named solvers for the common pairs of a scheme and an iteration.

A preset (ADR 0004) is an iteration whose scheme is built from flat
parameters, so the simple call ``FDPoisson(n=64).solve(problem)`` stays and
its :attr:`~maesolvers.core.solver.Solver.hparams` stay flat. The presets are
what :data:`maesolvers.registry.SOLVERS` and the comparison tables list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maesolvers.classical.iterations import Newton, PoissonIteration
from maesolvers.classical.schemes import Standard9


@dataclass(frozen=True, kw_only=True)
class FDNewton(Newton):
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

    scheme: Standard9 = field(init=False, repr=False)
    n: int = 64
    spline_degree: int = 3

    def __post_init__(self) -> None:
        scheme = Standard9(n=self.n, spline_degree=self.spline_degree)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", scheme.n)
        object.__setattr__(self, "spline_degree", scheme.spline_degree)
        super().__post_init__()


@dataclass(frozen=True, kw_only=True)
class FDPoisson(PoissonIteration):
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

    scheme: Standard9 = field(init=False, repr=False)
    n: int = 64
    spline_degree: int = 3

    def __post_init__(self) -> None:
        scheme = Standard9(n=self.n, spline_degree=self.spline_degree)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", scheme.n)
        object.__setattr__(self, "spline_degree", scheme.spline_degree)
        super().__post_init__()
