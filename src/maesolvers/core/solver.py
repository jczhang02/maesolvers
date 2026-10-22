"""The solver lifecycle (ADR 0003).

Every solver subclasses :class:`Solver`, whose :meth:`Solver.solve` is fixed:
``check -> setup -> run -> finalize``. A method author writes the hooks; the
base class times the solve, calls the callbacks, records the hyperparameters
and builds the :class:`~maesolvers.core.result.Result`, so every method
reports the same fields. :class:`IterativeSolver` fills in ``run`` for the
classical iterations.

Solvers are frozen dataclasses: their fields are the hyperparameters, and a
solve keeps its working data in the context returned by ``setup``, never on
the solver.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from maesolvers.core.callbacks import Callback
from maesolvers.core.domains import Field, HessianField
from maesolvers.core.problem import Problem
from maesolvers.core.result import Result

ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")

_RUN_KEYS = ("time", "iterations", "converged", "history", "unknowns")


class UnsupportedProblemError(ValueError):
    """The solver does not handle the problem's domain, boundary or dimension."""


@dataclass(frozen=True)
class Trace(Generic[StateT]):
    """What :meth:`Solver.run` returns.

    Parameters
    ----------
    state
        The final state (an iterate, a trained network), passed to
        :meth:`Solver.finalize`.
    iterations
        Steps taken.
    converged
        Whether the stopping criterion was met.
    history
        The convergence measure after every step.
    """

    state: StateT
    iterations: int
    converged: bool
    history: tuple[float, ...] = ()


@dataclass(frozen=True, eq=False)
class Solution:
    """What :meth:`Solver.finalize` returns: the method's part of the result.

    Parameters
    ----------
    u, hessian, nodes, values, node_hessian
        As in :class:`~maesolvers.core.result.Result`.
    unknowns
        Number of unknowns (degrees of freedom, network parameters), or None.
    info
        Method-specific entries for ``Result.info``; must not use the keys
        the base class sets (``time``, ``iterations``, ``converged``,
        ``history``, ``unknowns``).
    """

    u: Field
    hessian: HessianField | None = None
    nodes: np.ndarray | None = None
    values: np.ndarray | None = None
    node_hessian: np.ndarray | None = None
    unknowns: int | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


def _config(obj: object) -> Any:
    """The configuration of a dataclass part, nested parts included.

    Parameters
    ----------
    obj
        A dataclass instance such as a solver, or a plain value.

    Returns
    -------
    Any
        ``{"class_path": ..., "init_args": {...}}`` for a dataclass instance,
        built recursively over its init fields, the value itself otherwise;
        the form jsonargparse reads (ADR 0005).
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        return obj
    cls = type(obj)
    return {
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
        "init_args": {
            f.name: _config(getattr(obj, f.name)) for f in fields(obj) if f.init
        },
    }


class Solver(ABC, Generic[ContextT, StateT]):
    """Base class of every solver.

    A subclass is a frozen dataclass whose fields are its hyperparameters. It
    declares the problems it supports in the class variables ``domains``,
    ``boundaries`` and, if it is limited in dimension, ``dims``, and
    implements :meth:`setup`, :meth:`run` and :meth:`finalize`.

    Attributes
    ----------
    domains : tuple of type
        Domain classes the solver handles.
    boundaries : tuple of type
        Boundary condition classes the solver handles.
    dims : tuple of int or None
        Space dimensions the solver handles, None for any.
    """

    domains: ClassVar[tuple[type, ...]]
    boundaries: ClassVar[tuple[type, ...]]
    dims: ClassVar[tuple[int, ...] | None] = None

    @classmethod
    def supports(cls, problem: Problem) -> bool:
        """Whether the solver handles the problem's domain, boundary and dimension.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        if not hasattr(cls, "domains") or not hasattr(cls, "boundaries"):
            raise TypeError(f"{cls.__name__} must declare domains and boundaries")
        return (
            isinstance(problem.domain, cls.domains)
            and isinstance(problem.boundary, cls.boundaries)
            and (cls.dims is None or problem.dim in cls.dims)
        )

    def check(self, problem: Problem) -> None:
        """Raise if the solver cannot solve the problem.

        Override to add checks, and call ``super().check(problem)``.

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        """
        if not self.supports(problem):
            dims = "any dimension" if self.dims is None else f"dimensions {self.dims}"
            raise UnsupportedProblemError(
                f"{type(self).__name__} handles domains "
                f"{[c.__name__ for c in self.domains]}, boundaries "
                f"{[c.__name__ for c in self.boundaries]} in {dims}; problem "
                f"{problem.name} has a {problem.dim}D {type(problem.domain).__name__} "
                f"with a {type(problem.boundary).__name__} boundary"
            )

    @abstractmethod
    def setup(self, problem: Problem) -> ContextT:
        """Build the working data of one solve: grid, operators, network.

        Parameters
        ----------
        problem
            A problem that passed :meth:`check`.

        Returns
        -------
        ContextT
            Passed to :meth:`run` and :meth:`finalize`.
        """

    @abstractmethod
    def run(
        self, problem: Problem, context: ContextT, callbacks: Sequence[Callback]
    ) -> Trace[StateT]:
        """Compute the solution; call ``on_step_end`` after every step.

        Parameters
        ----------
        problem
            The problem.
        context
            What :meth:`setup` returned.
        callbacks
            The callbacks of this solve.

        Returns
        -------
        Trace
        """

    @abstractmethod
    def finalize(self, problem: Problem, context: ContextT, state: StateT) -> Solution:
        """Turn the final state into the solution fields of the result.

        Parameters
        ----------
        problem
            The problem.
        context
            What :meth:`setup` returned.
        state
            ``Trace.state`` from :meth:`run`.

        Returns
        -------
        Solution
        """

    @property
    def hparams(self) -> dict[str, Any]:
        """The solver's configuration, ``{"class_path": ..., "init_args": ...}``."""
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass")
        return _config(self)

    def solve(self, problem: Problem, callbacks: Sequence[Callback] = ()) -> Result:
        """Solve a problem: ``check -> setup -> run -> finalize``.

        Do not override; implement the hooks instead.

        Parameters
        ----------
        problem
            The problem.
        callbacks
            Observers of this solve.

        Returns
        -------
        Result
            With ``info`` holding ``time`` (seconds, setup to finalize),
            ``iterations``, ``converged``, ``history``, ``unknowns`` when known,
            and the method's own entries.
        """
        self.check(problem)
        config = self.hparams
        for callback in callbacks:
            callback.on_solve_start(self, problem)
        start = time.perf_counter()
        context = self.setup(problem)
        trace = self.run(problem, context, callbacks)
        solution = self.finalize(problem, context, trace.state)
        elapsed = time.perf_counter() - start
        clash = set(solution.info) & set(_RUN_KEYS)
        if clash:
            raise ValueError(f"Solution.info must not set {sorted(clash)}")
        info: dict[str, Any] = {
            "time": elapsed,
            "iterations": trace.iterations,
            "converged": trace.converged,
            "history": trace.history,
        }
        if solution.unknowns is not None:
            info["unknowns"] = solution.unknowns
        info.update(solution.info)
        result = Result(
            problem=problem,
            u=solution.u,
            hessian=solution.hessian,
            nodes=solution.nodes,
            values=solution.values,
            node_hessian=solution.node_hessian,
            info=info,
            hparams=config,
        )
        for callback in callbacks:
            callback.on_solve_end(self, result)
        return result


IterateT = TypeVar("IterateT")


@dataclass(frozen=True, kw_only=True)
class IterativeSolver(Solver[ContextT, IterateT]):
    """A solver that repeats one step until a convergence measure is small.

    Subclasses implement :meth:`initial_guess`, :meth:`step` and
    :meth:`converged` besides :meth:`Solver.setup` and
    :meth:`Solver.finalize`, set the class variable ``measure`` to the name
    of what :meth:`step` measures, and call ``super().__post_init__()`` from
    their own ``__post_init__``.

    Parameters
    ----------
    max_iter
        Most steps to take, at least 1.
    """

    max_iter: int = 100
    measure: ClassVar[str] = "measure"

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")

    @abstractmethod
    def initial_guess(self, problem: Problem, context: ContextT) -> IterateT:
        """The first iterate.

        Parameters
        ----------
        problem
            The problem.
        context
            What :meth:`Solver.setup` returned.

        Returns
        -------
        IterateT
        """

    @abstractmethod
    def step(
        self, context: ContextT, iterate: IterateT
    ) -> tuple[IterateT, float] | None:
        """One step from ``iterate``.

        Parameters
        ----------
        context
            What :meth:`Solver.setup` returned.
        iterate
            The current iterate; do not modify it.

        Returns
        -------
        tuple or None
            The next iterate and its convergence measure, or None when no step
            makes progress (the solve then stops unconverged).
        """

    @abstractmethod
    def converged(self, context: ContextT, measure: float) -> bool:
        """Whether a step's convergence measure meets the stopping criterion.

        Parameters
        ----------
        context
            What :meth:`Solver.setup` returned.
        measure
            The measure :meth:`step` returned.

        Returns
        -------
        bool
        """

    def run(
        self, problem: Problem, context: ContextT, callbacks: Sequence[Callback]
    ) -> Trace[IterateT]:
        """Step from :meth:`initial_guess` until converged, stalled or ``max_iter``.

        Parameters
        ----------
        problem
            The problem.
        context
            What :meth:`Solver.setup` returned.
        callbacks
            Each gets ``on_step_end`` with ``{measure: value}`` after every step.

        Returns
        -------
        Trace
        """
        iterate = self.initial_guess(problem, context)
        history: list[float] = []
        converged = False
        while len(history) < self.max_iter:
            stepped = self.step(context, iterate)
            if stepped is None:
                break
            iterate, measure = stepped
            history.append(float(measure))
            for callback in callbacks:
                callback.on_step_end(self, len(history), {self.measure: float(measure)})
            if self.converged(context, measure):
                converged = True
                break
        return Trace(iterate, len(history), converged, tuple(history))
