"""The Poisson iteration of Benamou, Froese and Oberman (2010), method (M2).

Same standard 9-point discretisation as :class:`FDNewton`, solved to its
fixed point by the iteration

    laplace u^{k+1} = sqrt((u^k_xx)^2 + (u^k_yy)^2 + 2 (u^k_xy)^2 + 2 f),

with Dirichlet data, started from ``laplace u^0 = sqrt(2 f)``.
The right-hand side is the Laplacian of a convex solution of the discrete
equation, so the iteration selects the convex branch where Newton's method
on the same discretisation may not (Froese-Oberman 2011, Section 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box
from maesolvers.core.problem import Problem
from maesolvers.core.solver import IterativeSolver, Solution
from maesolvers.solvers._grid import (
    Grid,
    build_grid,
    grid_parameters,
    grid_solution,
)


@dataclass(frozen=True, kw_only=True)
class FDPoisson(IterativeSolver[Grid, np.ndarray]):
    """Standard 9-point finite differences solved by the Poisson iteration.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001); at least 2.
    max_iter
        Most iterations.
    tol
        Stop when successive iterates differ by at most ``tol`` at every node.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the grid,
        1 to 5 and at most ``n``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...] | None] = (2,)
    measure: ClassVar[str] = "update"

    n: int = 64
    max_iter: int = 10_000
    tol: float = 1e-12
    spline_degree: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        n, tol, spline_degree = grid_parameters(self.n, self.tol, self.spline_degree)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "tol", tol)
        object.__setattr__(self, "spline_degree", spline_degree)

    def setup(self, problem: Problem) -> Grid:
        """Build the grid, the difference operators and the boundary data."""
        return build_grid(problem, self.n)

    def initial_guess(self, problem: Problem, context: Grid) -> np.ndarray:
        """Solve ``laplace u = sqrt(2 f)``."""
        return context.solve_poisson(np.sqrt(2.0 * context.f))

    def step(
        self, problem: Problem, context: Grid, iterate: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """One Poisson solve; the measure is the max change at the nodes."""
        uxx, uyy, uxy = context.second_derivatives(iterate)
        rhs = np.sqrt(uxx**2 + uyy**2 + 2.0 * uxy**2 + 2.0 * context.f)
        new = context.solve_poisson(rhs)
        return new, float(np.max(np.abs(new - iterate)))

    def converged(self, problem: Problem, context: Grid, measure: float) -> bool:
        """Successive iterates differ by at most ``tol``."""
        return measure <= self.tol

    def finalize(self, problem: Problem, context: Grid, state: np.ndarray) -> Solution:
        """Node values, the discrete Hessian and the spline interpolant."""
        return grid_solution(context, state, self.spline_degree)
