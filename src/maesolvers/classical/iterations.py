"""Iterations: the half of a classical solver that solves a scheme's equations.

An iteration is an :class:`~maesolvers.core.solver.IterativeSolver` with a
``scheme`` part (ADR 0004): ``Newton(scheme=Standard9(n=64))``. The scheme
decides which problems the solver supports; the iteration checks in
:meth:`~maesolvers.core.solver.Solver.check` that the scheme provides the
methods it calls, given by the protocols :class:`NewtonScheme` and
:class:`PoissonScheme`. The same scheme under two iterations can reach
different discrete solutions (``benchmarks/standard_fd.md``, Finding 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from maesolvers.classical.schemes import Scheme
from maesolvers.core._params import positive
from maesolvers.core.problem import Problem
from maesolvers.core.solver import IterativeSolver, Solution

_BACKTRACKS = 40
"""Halvings of the Newton step before the line search gives up."""

_SUFFICIENT_DECREASE = 1e-4
"""Armijo constant: a step of length ``t`` must cut the max residual by ``1e-4 t``."""


class NewtonScheme(Scheme, Protocol):
    """A scheme :class:`Newton` can solve: a residual, its Jacobian, a Laplace solve."""

    def residual(self, grid: Any, v: np.ndarray) -> np.ndarray:
        """The discrete equation's residual at the unknowns ``v``."""
        ...

    def jacobian(self, grid: Any, v: np.ndarray) -> sp.spmatrix:
        """The Jacobian of :meth:`residual` at ``v``."""
        ...

    def solve_laplace(self, grid: Any, rhs: np.ndarray) -> np.ndarray:
        """The unknowns of ``laplace u = rhs`` with the boundary data."""
        ...


class PoissonScheme(Scheme, Protocol):
    """A scheme :class:`PoissonIteration` can solve: second differences, a Laplace solve."""

    def second_derivatives(
        self, grid: Any, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``u_xx``, ``u_yy``, ``u_xy`` at the unknowns."""
        ...

    def solve_laplace(self, grid: Any, rhs: np.ndarray) -> np.ndarray:
        """The unknowns of ``laplace u = rhs`` with the boundary data."""
        ...


def _require(iteration: object, scheme: object, protocol: type) -> None:
    """Raise TypeError naming the methods of ``protocol`` that ``scheme`` lacks."""
    missing = [
        name
        for name in dir(protocol)
        if not name.startswith("_") and not callable(getattr(scheme, name, None))
    ]
    if missing:
        raise TypeError(
            f"{type(iteration).__name__} needs a scheme with {', '.join(missing)}; "
            f"{type(scheme).__name__} lacks them"
        )


@dataclass(frozen=True, kw_only=True)
class Newton(IterativeSolver[Any, np.ndarray]):
    """Damped Newton on a scheme's residual, from a Poisson start.

    Starts from ``laplace u = 2 sqrt(f)``, a lower bound for the Laplacian of
    the convex solution (Loeper-Rapetti 2005), and takes Newton steps halved
    until the max residual decreases. Nothing in the iteration prefers the
    convex root, so on a non-monotone scheme it may converge to a non-convex
    one.

    Parameters
    ----------
    scheme
        A :class:`NewtonScheme`.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``.
    """

    measure: ClassVar[str] = "residual"

    scheme: NewtonScheme
    max_iter: int = 50
    tol: float = 1e-10

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "tol", positive("tol", self.tol))

    def supports(self, problem: Problem) -> bool:
        """Whether the scheme supports the problem."""
        return self.scheme.supports(problem)

    def check(self, problem: Problem) -> None:
        """Raise if the scheme lacks a Newton method or does not support the problem.

        Raises
        ------
        TypeError
            If the scheme is not a :class:`NewtonScheme`.
        UnsupportedProblemError
            If the scheme does not support the problem.
        """
        _require(self, self.scheme, NewtonScheme)
        super().check(problem)

    def setup(self, problem: Problem) -> Any:
        """Discretise the problem with the scheme."""
        return self.scheme.discretize(problem)

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = 2 sqrt(f)``, the convex branch's trace."""
        return self.scheme.solve_laplace(
            context, 2.0 * np.sqrt(self.scheme.rhs(context))
        )

    def step(
        self, problem: Problem, context: Any, iterate: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """One Newton step, halved until the max residual decreases.

        An iterate that already meets the stopping criterion (a warm start)
        is returned unchanged with its residual. Returns None when no step
        length decreases the residual (rounding level).
        """
        r = self.scheme.residual(context, iterate)
        norm = float(np.max(np.abs(r)))
        if self.converged(problem, context, norm):
            return iterate, norm
        direction = spsolve(self.scheme.jacobian(context, iterate), -r)
        t = 1.0
        for _ in range(_BACKTRACKS):
            candidate = iterate + t * direction
            new_norm = float(np.max(np.abs(self.scheme.residual(context, candidate))))
            if new_norm < (1.0 - _SUFFICIENT_DECREASE * t) * norm:
                return candidate, new_norm
            t *= 0.5
        return None

    def converged(self, problem: Problem, context: Any, measure: float) -> bool:
        """Max residual at most ``tol * max(1, max |f|)``."""
        scale = max(1.0, float(np.max(np.abs(self.scheme.rhs(context)))))
        return measure <= self.tol * scale

    def finalize(self, problem: Problem, context: Any, state: np.ndarray) -> Solution:
        """The scheme's solution fields for the final iterate."""
        return self.scheme.solution(context, state)


@dataclass(frozen=True, kw_only=True)
class PoissonIteration(IterativeSolver[Any, np.ndarray]):
    """The Poisson iteration (M2) of Benamou, Froese and Oberman (2010).

    Iterates

        laplace u^{k+1} = sqrt((u^k_xx)^2 + (u^k_yy)^2 + 2 (u^k_xy)^2 + 2 f)

    with the boundary data, from ``laplace u^0 = sqrt(2 f)``. The right-hand
    side is the Laplacian of a convex solution of the discrete equation, so
    the iteration selects the convex branch where Newton's method on the same
    scheme may not (Froese-Oberman 2011, Section 8.1).

    Parameters
    ----------
    scheme
        A :class:`PoissonScheme`.
    max_iter
        Most iterations.
    tol
        Stop when successive iterates differ by at most ``tol`` at every node.
    """

    measure: ClassVar[str] = "update"

    scheme: PoissonScheme
    max_iter: int = 10_000
    tol: float = 1e-12

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "tol", positive("tol", self.tol))

    def supports(self, problem: Problem) -> bool:
        """Whether the scheme supports the problem."""
        return self.scheme.supports(problem)

    def check(self, problem: Problem) -> None:
        """Raise if the scheme lacks a method the iteration calls or the problem.

        Raises
        ------
        TypeError
            If the scheme is not a :class:`PoissonScheme`.
        UnsupportedProblemError
            If the scheme does not support the problem.
        """
        _require(self, self.scheme, PoissonScheme)
        super().check(problem)

    def setup(self, problem: Problem) -> Any:
        """Discretise the problem with the scheme."""
        return self.scheme.discretize(problem)

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = sqrt(2 f)``."""
        return self.scheme.solve_laplace(
            context, np.sqrt(2.0 * self.scheme.rhs(context))
        )

    def step(
        self, problem: Problem, context: Any, iterate: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """One Poisson solve; the measure is the max change at the nodes."""
        uxx, uyy, uxy = self.scheme.second_derivatives(context, iterate)
        f = self.scheme.rhs(context)
        new = self.scheme.solve_laplace(
            context, np.sqrt(uxx**2 + uyy**2 + 2.0 * uxy**2 + 2.0 * f)
        )
        return new, float(np.max(np.abs(new - iterate)))

    def converged(self, problem: Problem, context: Any, measure: float) -> bool:
        """Successive iterates differ by at most ``tol``."""
        return measure <= self.tol

    def finalize(self, problem: Problem, context: Any, state: np.ndarray) -> Solution:
        """The scheme's solution fields for the final iterate."""
        return self.scheme.solution(context, state)
