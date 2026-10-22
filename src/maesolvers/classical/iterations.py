"""Iterations: the half of a classical solver that solves a scheme's equations.

An iteration is a :class:`SchemeIteration`, an
:class:`~maesolvers.core.solver.IterativeSolver` with a
:class:`~maesolvers.classical.schemes.Scheme` part (ADR 0004):
``Newton(scheme=Standard9(n=64))``. The scheme decides which problems the
solver supports; the iteration names in :attr:`SchemeIteration.requires` the
methods it calls beyond the base four, and checks in
:meth:`~maesolvers.core.solver.Solver.check` that the scheme provides them.
The same scheme under two iterations can reach different discrete solutions
(``benchmarks/standard_fd.md``, Finding 2).
"""

# No `from __future__ import annotations` here. jsonargparse reads the field types
# (ADR 0005); with postponed annotations in this module, jsonargparse 4.52 fails to
# evaluate the dataclass-generated __init__ of the presets (AttributeError in
# _postponed_annotations.get_types) and leaves every parameter of FDNewton and
# FDPoisson, plain int and float included, Unvalidated. Guarded by
# test_every_numeric_solver_parameter_is_checked_against_its_type.

from dataclasses import dataclass, is_dataclass
from typing import Any, ClassVar, Protocol, cast

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


@dataclass(frozen=True, kw_only=True)
class SchemeIteration(IterativeSolver[Any, np.ndarray]):
    """Base class of an iteration that solves a scheme's discrete equations.

    The plug-in point for a new iteration: a subclass is decorated with
    ``@dataclass(frozen=True, kw_only=True)``, sets the class variables
    ``measure`` and :attr:`requires`, gives ``tol`` a default, and implements
    :meth:`~maesolvers.core.solver.IterativeSolver.initial_guess`,
    :meth:`~maesolvers.core.solver.IterativeSolver.step` and
    :meth:`~maesolvers.core.solver.IterativeSolver.converged`. The context of
    every hook is the scheme's grid and the iterate is the vector of
    unknowns; this class asks the scheme for the support, the grid and the
    solution fields.

    Parameters
    ----------
    scheme
        The scheme whose equations the iteration solves, an instance of a
        frozen-dataclass subclass of
        :class:`~maesolvers.classical.schemes.Scheme`.
    max_iter
        Most steps, an integer of at least 1.
    tol
        Positive tolerance of the subclass's stopping criterion.

    Attributes
    ----------
    requires : tuple of str
        Methods the iteration calls on the scheme beyond the four of
        :class:`~maesolvers.classical.schemes.Scheme`.

    Raises
    ------
    TypeError
        On construction, if ``scheme`` is not an instance of a frozen
        dataclass that subclasses :class:`~maesolvers.classical.schemes.Scheme`.
    ValueError
        On construction, if ``tol`` is not positive.
    """

    requires: ClassVar[tuple[str, ...]] = ()

    scheme: Scheme
    tol: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_scheme(self, self.scheme)
        object.__setattr__(self, "tol", positive("tol", self.tol))

    def supports(self, problem: Problem) -> bool:
        """Whether the scheme supports the problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return self.scheme.supports(problem)

    def check(self, problem: Problem) -> None:
        """Raise if the scheme lacks a required method or the problem.

        Parameters
        ----------
        problem
            The problem to solve.

        Raises
        ------
        TypeError
            If the scheme lacks a method named in :attr:`requires`.
        UnsupportedProblemError
            If the scheme does not support the problem.
        """
        missing = [
            name
            for name in self.requires
            if not callable(getattr(self.scheme, name, None))
        ]
        if missing:
            raise TypeError(
                f"{type(self).__name__} needs a scheme with {', '.join(missing)}, "
                f"which {type(self.scheme).__name__} does not have"
            )
        super().check(problem)

    def setup(self, problem: Problem) -> Any:
        """Discretise the problem with the scheme.

        Parameters
        ----------
        problem
            A problem that passed :meth:`check`.

        Returns
        -------
        Any
            The scheme's grid, the context of the other hooks.
        """
        return self.scheme.discretize(problem)

    def finalize(self, problem: Problem, context: Any, state: np.ndarray) -> Solution:
        """Return the scheme's solution fields for the final iterate.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.
        state
            The final unknowns.

        Returns
        -------
        Solution
        """
        return self.scheme.solution(context, state)


def _check_scheme(iteration: object, scheme: object) -> None:
    """Raise TypeError unless ``scheme`` is a frozen-dataclass Scheme instance."""
    owner = f"{type(iteration).__name__}.scheme"
    if isinstance(scheme, type):
        raise TypeError(
            f"{owner} must be an instance, got the class {scheme.__name__}; "
            f"write {scheme.__name__}(...)"
        )
    cls = type(scheme)
    if not isinstance(scheme, Scheme):
        raise TypeError(f"{owner} must subclass Scheme, got {cls.__name__}")
    params = cls.__dict__.get("__dataclass_params__")
    if not is_dataclass(scheme) or params is None or not params.frozen:
        raise TypeError(
            f"{owner} must be a frozen dataclass instance, got {cls.__name__}; "
            "decorate it with @dataclass(frozen=True)"
        )


class _NewtonScheme(Protocol):
    """The scheme methods :class:`Newton` calls: ``rhs`` and its ``requires``."""

    def rhs(self, grid: Any) -> np.ndarray: ...
    def residual(self, grid: Any, v: np.ndarray) -> np.ndarray: ...
    def jacobian(self, grid: Any, v: np.ndarray) -> sp.spmatrix: ...
    def solve_laplace(self, grid: Any, rhs: np.ndarray) -> np.ndarray: ...


class _PoissonScheme(Protocol):
    """The scheme methods :class:`PoissonIteration` calls: ``rhs`` and its ``requires``."""

    def rhs(self, grid: Any) -> np.ndarray: ...
    def second_derivatives(
        self, grid: Any, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def solve_laplace(self, grid: Any, rhs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, kw_only=True)
class Newton(SchemeIteration):
    """Damped Newton on a scheme's residual, from a Poisson start.

    Starts from ``laplace u = 2 sqrt(f)``, a lower bound for the Laplacian of
    the convex solution (Loeper-Rapetti 2005), and takes Newton steps halved
    until the max residual decreases. Nothing in the iteration prefers the
    convex root, so on a non-monotone scheme it may converge to a non-convex
    one.

    Parameters
    ----------
    scheme
        A scheme with ``residual(grid, v)``, its sparse ``jacobian(grid, v)``
        and ``solve_laplace(grid, rhs)``.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``.
    """

    measure: ClassVar[str] = "residual"
    requires: ClassVar[tuple[str, ...]] = ("residual", "jacobian", "solve_laplace")

    max_iter: int = 50
    tol: float = 1e-10

    @property
    def _scheme(self) -> _NewtonScheme:
        """The scheme, typed by the methods :meth:`check` makes sure it has."""
        return cast(_NewtonScheme, self.scheme)

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = 2 sqrt(f)``, the convex branch's trace.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.

        Returns
        -------
        numpy.ndarray
            The unknowns.
        """
        return self._scheme.solve_laplace(
            context, 2.0 * np.sqrt(self._scheme.rhs(context))
        )

    def step(
        self, problem: Problem, context: Any, iterate: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """One Newton step, halved until the max residual decreases.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.
        iterate
            The current unknowns.

        Returns
        -------
        tuple or None
            The next unknowns and their max residual. An iterate that already
            meets the stopping criterion (a warm start) is returned unchanged.
            None when no step length decreases the residual (rounding level).
        """
        r = self._scheme.residual(context, iterate)
        norm = float(np.max(np.abs(r)))
        if self.converged(problem, context, norm):
            return iterate, norm
        direction = spsolve(self._scheme.jacobian(context, iterate), -r)
        t = 1.0
        for _ in range(_BACKTRACKS):
            candidate = iterate + t * direction
            new_norm = float(np.max(np.abs(self._scheme.residual(context, candidate))))
            if new_norm < (1.0 - _SUFFICIENT_DECREASE * t) * norm:
                return candidate, new_norm
            t *= 0.5
        return None

    def converged(self, problem: Problem, context: Any, measure: float) -> bool:
        """Whether the max residual is at most ``tol * max(1, max |f|)``.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.
        measure
            The max residual.

        Returns
        -------
        bool
        """
        scale = max(1.0, float(np.max(np.abs(self._scheme.rhs(context)))))
        return measure <= self.tol * scale


@dataclass(frozen=True, kw_only=True)
class PoissonIteration(SchemeIteration):
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
        A scheme with ``second_derivatives(grid, v)`` and
        ``solve_laplace(grid, rhs)``.
    max_iter
        Most iterations.
    tol
        Stop when successive iterates differ by at most ``tol`` at every node.
    """

    measure: ClassVar[str] = "update"
    requires: ClassVar[tuple[str, ...]] = ("second_derivatives", "solve_laplace")

    max_iter: int = 10_000
    tol: float = 1e-12

    @property
    def _scheme(self) -> _PoissonScheme:
        """The scheme, typed by the methods :meth:`check` makes sure it has."""
        return cast(_PoissonScheme, self.scheme)

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = sqrt(2 f)``.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.

        Returns
        -------
        numpy.ndarray
            The unknowns.
        """
        return self._scheme.solve_laplace(
            context, np.sqrt(2.0 * self._scheme.rhs(context))
        )

    def step(
        self, problem: Problem, context: Any, iterate: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """One Poisson solve.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.
        iterate
            The current unknowns.

        Returns
        -------
        tuple
            The next unknowns and the max change at the nodes.
        """
        uxx, uyy, uxy = self._scheme.second_derivatives(context, iterate)
        f = self._scheme.rhs(context)
        new = self._scheme.solve_laplace(
            context, np.sqrt(uxx**2 + uyy**2 + 2.0 * uxy**2 + 2.0 * f)
        )
        return new, float(np.max(np.abs(new - iterate)))

    def converged(self, problem: Problem, context: Any, measure: float) -> bool:
        """Whether successive iterates differ by at most ``tol``.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.
        measure
            The max change at the nodes.

        Returns
        -------
        bool
        """
        return measure <= self.tol
