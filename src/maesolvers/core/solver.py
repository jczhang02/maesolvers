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
from dataclasses import KW_ONLY, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from maesolvers.core._params import integer
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
        As in :class:`~maesolvers.core.result.Result`; all but ``u`` are
        keyword-only.
    unknowns
        Number of unknowns (degrees of freedom, network parameters), or None.
    info
        Method-specific entries for ``Result.info``; must not use the keys
        the base class sets (``time``, ``iterations``, ``converged``,
        ``history``, ``unknowns``).
    """

    u: Field
    _: KW_ONLY
    hessian: HessianField | None = None
    nodes: np.ndarray | None = None
    values: np.ndarray | None = None
    node_hessian: np.ndarray | None = None
    unknowns: int | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


def _config(obj: object) -> Any:
    """The configuration of a part, nested parts included.

    Parameters
    ----------
    obj
        A dataclass instance such as a solver, a tuple, list or mapping of
        parts, or a plain value.

    Returns
    -------
    Any
        ``{"class_path": ..., "init_args": {...}}`` for a dataclass instance,
        built recursively over its init fields; a tuple, list or dict of the
        configurations of the elements for a container; the value itself
        otherwise. This is the form jsonargparse reads (ADR 0005).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        return {
            "class_path": f"{cls.__module__}.{cls.__qualname__}",
            "init_args": {
                f.name: _config(getattr(obj, f.name)) for f in fields(obj) if f.init
            },
        }
    if isinstance(obj, tuple):
        return tuple(_config(v) for v in obj)
    if isinstance(obj, list):
        return [_config(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: _config(v) for k, v in obj.items()}
    return obj


class Solver(ABC, Generic[ContextT, StateT]):
    """Base class of every solver.

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters. It declares the problems it supports
    in the class variables ``domains``, ``boundaries`` and, if it is limited
    in dimension, ``dims``, and implements :meth:`setup`, :meth:`run` and
    :meth:`finalize`. A solver composed of parts (ADR 0004) may instead
    override :meth:`supports` and ask its parts. A subclass that defines
    ``__post_init__`` calls ``super().__post_init__()`` from it.

    Attributes
    ----------
    domains : tuple of type
        Domain classes the solver handles.
    boundaries : tuple of type
        Boundary condition classes the solver handles.
    dims : tuple of int or None
        Space dimensions the solver handles, None for any.

    Raises
    ------
    TypeError
        On construction, or at the latest in :meth:`solve`, if the class
        itself is not decorated as a frozen dataclass (an inherited
        decoration does not count: its new fields would miss from
        :attr:`hparams`).
    """

    domains: ClassVar[tuple[type, ...]]
    boundaries: ClassVar[tuple[type, ...]]
    dims: ClassVar[tuple[int, ...] | None] = None

    def __post_init__(self) -> None:
        """Check the class definition; a subclass's own ``__post_init__`` calls it."""
        self._check_definition()

    @classmethod
    def _check_definition(cls) -> None:
        """Raise TypeError if the class misses a declaration its base needs."""
        params = cls.__dict__.get("__dataclass_params__")
        if params is None or not params.frozen:
            raise TypeError(
                f"{cls.__name__} must be decorated with @dataclass(frozen=True)"
            )

    def supports(self, problem: Problem) -> bool:
        """Whether the solver handles the problem's domain, boundary and dimension.

        The default reads the class variables ``domains``, ``boundaries`` and
        ``dims``.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        cls = type(self)
        if not hasattr(cls, "domains") or not hasattr(cls, "boundaries"):
            raise TypeError(
                f"{cls.__name__} must declare domains and boundaries "
                "or override supports"
            )
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
            raise UnsupportedProblemError(
                f"{type(self).__name__} does not support {problem.name}, a "
                f"{problem.dim}D {type(problem.domain).__name__} problem with a "
                f"{type(problem.boundary).__name__} boundary"
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
        """The solver's configuration, ``{"class_path": ..., "init_args": ...}``.

        Nested parts are configurations of the same form, so
        ``type(s)(**s.hparams["init_args"]) == s`` for a solver without parts.
        """
        self._check_definition()
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
        config = self.hparams
        self.check(problem)
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
    :meth:`Solver.finalize` and set the class variable ``measure`` to the
    name of what :meth:`step` measures (the key of ``on_step_end`` metrics).
    Every hook after :meth:`Solver.setup` takes the problem and the context
    first.

    Parameters
    ----------
    max_iter
        Most steps to take, an integer of at least 1.
    """

    max_iter: int = 100
    measure: ClassVar[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "max_iter", integer("max_iter", self.max_iter, 1))

    @classmethod
    def _check_definition(cls) -> None:
        super()._check_definition()
        if not isinstance(getattr(cls, "measure", None), str):
            raise TypeError(
                f"{cls.__name__} must set the class variable measure, the name "
                "of what step measures"
            )

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
        self, problem: Problem, context: ContextT, iterate: IterateT
    ) -> tuple[IterateT, float] | None:
        """One step from ``iterate``.

        Parameters
        ----------
        problem
            The problem.
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
    def converged(self, problem: Problem, context: ContextT, measure: float) -> bool:
        """Whether a step's convergence measure meets the stopping criterion.

        Parameters
        ----------
        problem
            The problem.
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
            stepped = self.step(problem, context, iterate)
            if stepped is None:
                break
            iterate, measure = stepped
            history.append(float(measure))
            for callback in callbacks:
                callback.on_step_end(self, len(history), {self.measure: float(measure)})
            if self.converged(problem, context, measure):
                converged = True
                break
        return Trace(iterate, len(history), converged, tuple(history))
