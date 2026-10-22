"""Hooks into a solve, the same for every solver family (ADR 0003)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from maesolvers.core.problem import Problem
from maesolvers.core.result import Result

if TYPE_CHECKING:
    from maesolvers.core.solver import Solver


class Callback:
    """Base class for callbacks; override the hooks you need.

    A callback observes a solve and must not change the solver, the problem
    or the iterate. :meth:`Solver.solve <maesolvers.core.solver.Solver.solve>`
    calls the hooks in the order start, step end for every step, end.
    """

    def on_solve_start(self, solver: Solver, problem: Problem) -> None:
        """Called after the problem passed the solver's check, before setup.

        Parameters
        ----------
        solver
            The running solver.
        problem
            The problem it solves.
        """

    def on_step_end(
        self, solver: Solver, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Called after every iteration or training step.

        Parameters
        ----------
        solver
            The running solver.
        step
            Number of completed steps, from 1.
        metrics
            What the step measured, such as ``{"residual": 3e-7}``.
        """

    def on_solve_end(self, solver: Solver, result: Result) -> None:
        """Called with the finished result.

        Parameters
        ----------
        solver
            The solver that ran.
        result
            Its result.
        """
