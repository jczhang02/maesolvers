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

import dataclasses
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, spsolve

from maesolvers.classical.schemes import Scheme
from maesolvers.core._params import positive
from maesolvers.core.callbacks import Callback
from maesolvers.core.problem import Problem
from maesolvers.core.solver import IterativeSolver, Solution, Trace, check_part

_BACKTRACKS = 40
"""Halvings of the Newton step before the line search gives up."""

_SUFFICIENT_DECREASE = 1e-4
"""Armijo constant: a step of length ``t`` must cut the max residual by ``1e-4 t``."""

_ILU_DROP_TOL = 1e-4
"""Drop tolerance of the incomplete LU that preconditions ``PolicyIteration``'s BiCGSTAB."""

_ILU_FILL = 10.0
"""Fill factor of that incomplete LU: at most ten times the matrix's nonzeros."""


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
        On construction, if ``scheme`` is not an instance of a subclass of
        :class:`~maesolvers.classical.schemes.Scheme` that carries its own
        ``@dataclass(frozen=True)`` decorator and takes keyword arguments
        only.
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
    """Raise TypeError unless ``scheme`` is an instance of a well-defined Scheme.

    The scheme's class must pass the check a solver class passes: its own
    frozen dataclass decorator and keyword-only fields.
    """
    check_part(f"{type(iteration).__name__}.scheme", scheme, Scheme)


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
    """Damped Newton on a scheme's residual, from a Poisson start or another solver's iterate.

    Starts from ``laplace u = d f^(1/d)`` in dimension ``d`` (``2 sqrt(f)``
    in 2D), a lower bound for the Laplacian of the convex solution by the
    inequality of the arithmetic and geometric means of the Hessian's
    eigenvalues (Loeper-Rapetti 2005 in 2D), or, if ``start`` is given,
    from the final iterate of that solver, and takes Newton steps halved
    until the max residual decreases. Nothing in the iteration prefers the
    convex root, so on a non-monotone scheme it may converge to a non-convex
    one; a start on a monotone scheme, such as
    :class:`~maesolvers.classical.presets.FDMonotone`, puts it next to the
    convex root (Froese-Oberman 2011, J. Comput. Phys., Section 7.1).

    Parameters
    ----------
    scheme
        A scheme with ``residual(grid, v)``, its sparse ``jacobian(grid, v)``
        and ``solve_laplace(grid, rhs)``.
    max_iter
        Most Newton steps on ``scheme``.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``, the residual
        as the scheme defines it (:meth:`Standard9.residual
        <maesolvers.classical.schemes.Standard9.residual>` scales it next to
        a curved boundary).
    start
        None, or a :class:`SchemeIteration` whose scheme has the same
        unknowns, such as ``FDMonotone(n)`` or
        ``PoissonIteration(scheme=Standard9(n), max_iter=1)``. :meth:`run`
        then runs ``start`` first, with its own ``max_iter`` and ``tol``
        (its ``setup`` and ``run``, not its ``finalize``), and steps from its
        final unknowns, converged or not, instead of calling
        :meth:`initial_guess`. ``Result.info["iterations"]``, ``converged``,
        ``history`` and the callbacks cover the steps on ``scheme`` only;
        ``info["start_iterations"]`` and ``info["start_converged"]`` report
        the start, and ``time`` includes it.

    Raises
    ------
    TypeError
        On construction, if ``start`` is neither None nor an instance of a
        well-defined subclass of :class:`SchemeIteration`.
    """

    measure: ClassVar[str] = "residual"
    requires: ClassVar[tuple[str, ...]] = ("residual", "jacobian", "solve_laplace")

    max_iter: int = 50
    tol: float = 1e-10
    start: SchemeIteration | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.start is not None:
            check_part(f"{type(self).__name__}.start", self.start, SchemeIteration)

    def supports(self, problem: Problem) -> bool:
        """Whether the scheme, and ``start`` if given, support the problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return super().supports(problem) and (
            self.start is None or self.start.supports(problem)
        )

    def check(self, problem: Problem) -> None:
        """Raise if the scheme lacks a required method, or a part the problem.

        Parameters
        ----------
        problem
            The problem to solve.

        Raises
        ------
        TypeError
            If the scheme lacks a method named in :attr:`requires`, or
            ``start.check`` raises it.
        UnsupportedProblemError
            If the scheme or ``start`` does not support the problem.
        """
        super().check(problem)
        if self.start is not None:
            self.start.check(problem)

    @property
    def _scheme(self) -> _NewtonScheme:
        """The scheme, typed by the methods :meth:`check` makes sure it has."""
        return cast(_NewtonScheme, self.scheme)

    def run(
        self, problem: Problem, context: Any, callbacks: Sequence[Callback]
    ) -> Trace[np.ndarray]:
        """Step from :meth:`initial_guess`, or from the final iterate of ``start``.

        Parameters
        ----------
        problem
            The problem.
        context
            The scheme's grid.
        callbacks
            Each gets ``on_step_end`` with ``{"residual": value}`` after every
            step on the scheme; none sees the steps of ``start``.

        Returns
        -------
        Trace
            With a ``start``, its ``info`` holds ``start_iterations`` and
            ``start_converged``.

        Raises
        ------
        ValueError
            If ``start`` ends with another number of unknowns than the scheme
            has.

        Warns
        -----
        RuntimeWarning
            If ``start`` did not converge; Newton steps from its last iterate.
        """
        if self.start is None:
            return super().run(problem, context, callbacks)
        begun = self.start.run(problem, self.start.setup(problem), ())
        unknowns = self._scheme.rhs(context).shape
        if np.shape(begun.state) != unknowns:
            raise ValueError(
                f"{type(self).__name__}.start has {np.size(begun.state)} unknowns "
                f"on {problem.name}, the scheme {unknowns[0]}"
            )
        if not begun.converged:
            warnings.warn(
                f"{type(self).__name__}.start did not converge on {problem.name} "
                f"in {begun.iterations} steps; Newton steps from its last iterate",
                RuntimeWarning,
                stacklevel=3,  # the caller of solve
            )
        trace = self.run_from(problem, context, callbacks, begun.state)
        info = {
            "start_iterations": begun.iterations,
            "start_converged": begun.converged,
        }
        return dataclasses.replace(trace, info={**trace.info, **info})

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = d f^(1/d)``, the convex branch's trace.

        In 2D ``laplace u = 2 sqrt(f)``. :meth:`run` calls it only if
        ``start`` is None.

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
        d = problem.dim
        # numpy takes the square root for ** 0.5, so 2D starts are 2 sqrt(f) exactly
        return self._scheme.solve_laplace(
            context, d * self._scheme.rhs(context) ** (1.0 / d)
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


class _PolicyScheme(Protocol):
    """The scheme methods :class:`PolicyIteration` calls: ``rhs`` and its ``requires``."""

    def rhs(self, grid: Any) -> np.ndarray: ...
    def policy_system(
        self, grid: Any, v: np.ndarray
    ) -> tuple[sp.spmatrix, np.ndarray]: ...
    def solve_laplace(self, grid: Any, rhs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, kw_only=True)
class PolicyIteration(SchemeIteration):
    """Policy iteration (Howard's algorithm) on a scheme in HJB form.

    For a scheme whose residual is a maximum over controls of linear
    operators, ``max over policies of A v - F``, each step finds the optimal
    policy at the current unknowns and solves its linear system,
    ``A v_new = F`` (Chen-Wan-Lin 2018, Section 5.1; Howard 1960). When the
    matrix of every policy is an M-matrix, as for a monotone scheme, the
    iteration converges from any start (Bokanowski, Maroso and Zidani 2009);
    it is Newton's method with full steps and the policy's matrix as the
    Jacobian.

    Starts from ``laplace u = 2 sqrt(f)``, the solution for the control
    ``a = 1/2`` (Chen-Wan-Lin 2018, eq. (50)).

    Parameters
    ----------
    scheme
        A scheme with ``policy_system(grid, v)``, which returns the sparse
        ``A`` and the vector ``F`` of the optimal policy at ``v``, and
        ``solve_laplace(grid, rhs)``.
    max_iter
        Most steps.
    tol
        Stop when ``max |A v - F| <= tol`` at the current unknowns, the
        paper's tolerance on the residual by default.
    krylov_tol
        None solves each policy's system by sparse LU. A positive value
        solves it by BiCGSTAB to that relative residual, preconditioned by
        an incomplete LU (drop tolerance ``1e-4``), and by LU if BiCGSTAB
        fails; the paper uses GMRES with an incomplete LU. Needed where the
        LU fills in, as for the 17-point wide stencil at ``n = 512``.

    Notes
    -----
    A step measures the residual at the unknowns it starts from, as
    Howard's algorithm does, and returns them unchanged once that residual
    is at most ``tol``. So ``Result.info["history"]`` holds the residual of
    each iterate before its update, the last one at most ``tol`` when the
    solve converged, and ``iterations`` counts the policies evaluated,
    the paper's "number of policy iterations".
    """

    measure: ClassVar[str] = "residual"
    requires: ClassVar[tuple[str, ...]] = ("policy_system", "solve_laplace")

    max_iter: int = 100
    tol: float = 1e-6
    krylov_tol: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.krylov_tol is not None:
            object.__setattr__(
                self, "krylov_tol", positive("krylov_tol", self.krylov_tol)
            )

    @property
    def _scheme(self) -> _PolicyScheme:
        """The scheme, typed by the methods :meth:`check` makes sure it has."""
        return cast(_PolicyScheme, self.scheme)

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = 2 sqrt(f)``.

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
        """Evaluate the optimal policy at ``iterate`` and solve its system.

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
            The next unknowns and the max residual at ``iterate``; ``iterate``
            itself once that residual is at most ``tol``. None if the solve
            of the policy's system is not finite.
        """
        matrix, rhs = self._scheme.policy_system(context, iterate)
        norm = float(np.max(np.abs(matrix @ iterate - rhs)))
        if self.converged(problem, context, norm):
            return iterate, norm
        new = self._solve(sp.csc_matrix(matrix), rhs, iterate)
        if not np.all(np.isfinite(new)):
            return None
        return new, norm

    def _solve(
        self, matrix: sp.csc_matrix, rhs: np.ndarray, guess: np.ndarray
    ) -> np.ndarray:
        """Solve the policy's system, by LU or by BiCGSTAB with an incomplete LU."""
        if self.krylov_tol is not None:
            # Natural order and diagonal pivots: the incomplete LU of an
            # M-matrix then has positive pivots whatever it drops.
            ilu = spilu(
                matrix,
                drop_tol=_ILU_DROP_TOL,
                fill_factor=_ILU_FILL,
                permc_spec="NATURAL",
                diag_pivot_thresh=0.0,
            )
            new, info = bicgstab(
                matrix,
                rhs,
                x0=guess,
                rtol=self.krylov_tol,
                atol=0.0,
                M=LinearOperator(matrix.shape, ilu.solve),
            )
            if info == 0:
                return np.asarray(new)
        return np.asarray(spsolve(matrix, rhs))

    def converged(self, problem: Problem, context: Any, measure: float) -> bool:
        """Whether the max residual is at most ``tol``.

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
        return measure <= self.tol
