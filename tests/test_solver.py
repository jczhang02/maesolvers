"""The solver lifecycle, the iterative loop, callbacks and hyperparameters."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from maesolvers import (
    P1,
    P3,
    P4,
    Box,
    Callback,
    Dirichlet,
    FDNewton,
    FDPoisson,
    IterativeSolver,
    Problem,
    Solution,
    Solver,
    UnsupportedProblemError,
)
from maesolvers.core import Disk, Trace


def _zero(x):
    return np.zeros(np.shape(x)[:-1])


class Recorder(Callback):
    def __init__(self):
        self.events = []

    def on_solve_start(self, solver, problem):
        self.events.append(("start", problem.name))

    def on_step_end(self, solver, step, metrics):
        self.events.append(("step", step, dict(metrics)))

    def on_solve_end(self, solver, result):
        self.events.append(("end", result.info["iterations"]))


@dataclass(frozen=True, kw_only=True)
class Halving(IterativeSolver[float, float]):
    """Halves a number until it is below ``tol``; stalls below ``floor``."""

    domains: ClassVar[tuple[type, ...]] = (Box, Disk)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    measure: ClassVar[str] = "size"

    tol: float = 0.1
    floor: float = 0.0

    def setup(self, problem):
        return 1.0

    def initial_guess(self, problem, context):
        return context

    def step(self, context, iterate):
        if iterate / 2 < self.floor:
            return None
        return iterate / 2, iterate / 2

    def converged(self, context, measure):
        return measure < self.tol

    def finalize(self, problem, context, state):
        return Solution(u=_zero, unknowns=1, info={"final": state})


def test_iterative_loop_stops_when_converged():
    recorder = Recorder()
    result = Halving().solve(P4, callbacks=[recorder])
    assert result.info["converged"] and result.info["iterations"] == 4
    assert result.info["history"] == (0.5, 0.25, 0.125, 0.0625)
    assert result.info["final"] == 0.0625 and result.info["unknowns"] == 1
    assert result.info["time"] >= 0.0
    assert result.problem is P4
    assert recorder.events == [
        ("start", "P4"),
        ("step", 1, {"size": 0.5}),
        ("step", 2, {"size": 0.25}),
        ("step", 3, {"size": 0.125}),
        ("step", 4, {"size": 0.0625}),
        ("end", 4),
    ]


def test_iterative_loop_stops_at_max_iter_unconverged():
    result = Halving(max_iter=2).solve(P4)
    assert not result.info["converged"] and result.info["iterations"] == 2


def test_iterative_loop_stops_unconverged_when_a_step_stalls():
    result = Halving(floor=0.2).solve(P4)
    assert not result.info["converged"]
    assert result.info["history"] == (0.5, 0.25)


def test_max_iter_must_be_positive():
    with pytest.raises(ValueError, match="max_iter"):
        Halving(max_iter=0)


def test_solvers_take_keyword_arguments_only():
    with pytest.raises(TypeError):
        FDNewton(64)


def test_solvers_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FDNewton().n = 32


def test_hparams_record_the_class_and_every_field():
    assert FDNewton(n=16).hparams == {
        "class_path": "maesolvers.solvers.fd_newton.FDNewton",
        "init_args": {"max_iter": 50, "n": 16, "tol": 1e-10, "spline_degree": 3},
    }


def test_result_carries_the_hparams():
    assert Halving(tol=0.3).solve(P4).hparams["init_args"] == {
        "max_iter": 100,
        "tol": 0.3,
        "floor": 0.0,
    }


def test_hparams_nest_dataclass_parts():
    @dataclass(frozen=True)
    class Part:
        k: int = 2

    @dataclass(frozen=True, kw_only=True)
    class Composite(Halving):
        part: Part = Part()

    assert Composite().hparams["init_args"]["part"] == {
        "class_path": f"{__name__}.{Part.__qualname__}",
        "init_args": {"k": 2},
    }


def test_supports_checks_domain_boundary_and_dimension():
    assert Halving.supports(P1) and Halving.supports(P4)
    assert not Halving.supports(P3)
    assert FDNewton.supports(P4) and not FDNewton.supports(P1)
    cube = Problem("C", Box((0,) * 3, (1,) * 3), _zero, Dirichlet(_zero))
    assert Halving.supports(cube) and not FDPoisson.supports(cube)


def test_unsupported_problem_is_rejected_before_any_hook():
    recorder = Recorder()
    with pytest.raises(UnsupportedProblemError, match="Transport"):
        Halving().solve(P3, callbacks=[recorder])
    assert recorder.events == []


def test_a_solver_must_declare_what_it_supports():
    class Undeclared(Solver):
        def setup(self, problem):
            return None

        def run(self, problem, context, callbacks):
            return Trace(None, 0, True)

        def finalize(self, problem, context, state):
            return Solution(u=_zero)

    with pytest.raises(TypeError, match="declare"):
        Undeclared().solve(P4)


def test_a_minimal_solver_plugs_in():
    @dataclass(frozen=True)
    class Exact(Solver[None, None]):
        domains: ClassVar[tuple[type, ...]] = (Box,)
        boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)

        def setup(self, problem):
            return None

        def run(self, problem, context, callbacks):
            return Trace(None, 0, True)

        def finalize(self, problem, context, state):
            return Solution(u=problem.exact)

    result = Exact().solve(P4)
    assert result.info["converged"] and "unknowns" not in result.info
    assert result.hparams["init_args"] == {}


def test_a_solver_that_is_not_a_dataclass_fails_before_setup():
    class Plain(Solver):
        domains = (Box,)
        boundaries = (Dirichlet,)

        def setup(self, problem):
            raise AssertionError("setup must not run")

        def run(self, problem, context, callbacks):
            raise AssertionError

        def finalize(self, problem, context, state):
            raise AssertionError

    with pytest.raises(TypeError, match="dataclass"):
        Plain().solve(P4)


def test_solution_info_may_not_override_run_fields():
    @dataclass(frozen=True, kw_only=True)
    class Clashing(Halving):
        def finalize(self, problem, context, state):
            return Solution(u=_zero, info={"time": 0.0})

    with pytest.raises(ValueError, match="time"):
        Clashing().solve(P4)


@pytest.mark.parametrize("solver", [FDNewton, FDPoisson])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(n=1), "n must be"),
        (dict(tol=0.0), "tol must be"),
        (dict(spline_degree=0), "spline_degree"),
        (dict(spline_degree=6), "spline_degree"),
        (dict(n=2, spline_degree=3), "spline_degree"),
        (dict(max_iter=0), "max_iter"),
    ],
)
def test_grid_solvers_validate_their_parameters(solver, kwargs, message):
    with pytest.raises(ValueError, match=message):
        solver(**kwargs)


@pytest.mark.parametrize("solver", [FDNewton, FDPoisson])
def test_grid_solvers_report_nodes_and_their_hessian(solver):
    recorder = Recorder()
    result = solver(n=8).solve(P4, callbacks=[recorder])
    assert result.nodes.shape == (81, 2) and result.values.shape == (81,)
    assert result.node_hessian.shape == (81, 2, 2)
    assert np.sum(np.isfinite(result.node_hessian[:, 0, 0])) == 49
    assert result.info["unknowns"] == 49 and result.info["h"] == (0.125, 0.125)
    steps = [e for e in recorder.events if e[0] == "step"]
    assert len(steps) == result.info["iterations"]
    assert set(steps[0][2]) == {solver.measure}
