"""Standard finite differences with a damped Newton iteration.

Centred second differences on a uniform grid over a 2D box, Dirichlet
boundary values imposed at the boundary nodes, Newton's method on the
discrete residual ``u_xx u_yy - u_xy^2 - f`` with backtracking, started from
the Poisson problem ``laplace u = 2 sqrt(f)`` (Loeper-Rapetti 2005; the
standard discretisation of Benamou-Froese-Oberman 2010).

The scheme is not monotone: on smooth problems it converges to the convex
solution from the Poisson start, but it carries no convexity guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box
from maesolvers.core.problem import Problem
from maesolvers.core.solver import IterativeSolver, Solution
from maesolvers.solvers._grid import (
    Grid,
    build_grid,
    check_grid_parameters,
    grid_solution,
)

_BACKTRACKS = 40
"""Halvings of the Newton step before the line search gives up."""


@dataclass(frozen=True, kw_only=True)
class FDNewton(IterativeSolver[Grid, np.ndarray]):
    """Standard 9-point finite differences solved by damped Newton.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max f)``.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the grid,
        1 to 5 and at most ``n``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...] | None] = (2,)
    measure: ClassVar[str] = "residual"

    n: int = 64
    max_iter: int = 50
    tol: float = 1e-10
    spline_degree: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        check_grid_parameters(self.n, self.tol, self.spline_degree)

    def setup(self, problem: Problem) -> Grid:
        """Build the grid, the difference operators and the boundary data."""
        return build_grid(problem, self.n)

    def initial_guess(self, problem: Problem, context: Grid) -> np.ndarray:
        """Solve ``laplace u = 2 sqrt(f)``, the convex branch's trace."""
        return context.solve_poisson(2.0 * np.sqrt(context.f))

    def step(
        self, context: Grid, iterate: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """One Newton step, halved until the max residual decreases.

        Returns None when no step length decreases it (rounding level).
        """
        uxx, uyy, uxy = context.second_derivatives(iterate)
        r = uxx * uyy - uxy**2 - context.f
        norm = float(np.max(np.abs(r)))
        jac = (
            sp.diags(uyy) @ context.axx
            + sp.diags(uxx) @ context.ayy
            - 2.0 * sp.diags(uxy) @ context.axy
        ).tocsc()
        direction = spsolve(jac, -r)
        t = 1.0
        for _ in range(_BACKTRACKS):
            candidate = iterate + t * direction
            new_norm = float(np.max(np.abs(context.residual(candidate))))
            if new_norm < (1.0 - 1e-4 * t) * norm:
                return candidate, new_norm
            t *= 0.5
        return None

    def converged(self, context: Grid, measure: float) -> bool:
        """Max residual at most ``tol * max(1, max f)``."""
        return measure <= self.tol * max(1.0, float(np.max(np.abs(context.f))))

    def finalize(self, problem: Problem, context: Grid, state: np.ndarray) -> Solution:
        """Node values, the discrete Hessian and the spline interpolant."""
        return grid_solution(context, state, self.spline_degree)
