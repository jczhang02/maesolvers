"""Iterations composed with a scheme, the presets, and the registry (ADR 0004)."""

import dataclasses
import json
import re
from typing import ClassVar

import numpy as np
import pytest

from maesolvers import (
    SOLVERS,
    T1,
    T2,
    T8,
    FDNewton,
    FDPoisson,
    Newton,
    PoissonIteration,
    Scheme,
    SchemeIteration,
    Standard9,
    UnsupportedProblemError,
)
from maesolvers.metrics import evaluate


@dataclasses.dataclass(frozen=True, kw_only=True)
class PoissonOnly(Scheme):
    """A user scheme with what PoissonIteration needs, and no Jacobian."""

    n: int = 8

    def _inner(self):
        return Standard9(n=self.n)

    def supports(self, problem):
        return self._inner().supports(problem)

    def discretize(self, problem):
        return self._inner().discretize(problem)

    def rhs(self, grid):
        return self._inner().rhs(grid)

    def solution(self, grid, v):
        return self._inner().solution(grid, v)

    def second_derivatives(self, grid, v):
        return self._inner().second_derivatives(grid, v)

    def solve_laplace(self, grid, rhs):
        return self._inner().solve_laplace(grid, rhs)


@pytest.mark.parametrize(
    ("preset", "iteration"), [(FDNewton, Newton), (FDPoisson, PoissonIteration)]
)
@pytest.mark.parametrize("problem", [T1, T2])
def test_a_preset_is_its_composition(preset, iteration, problem):
    composed = iteration(scheme=Standard9(n=16, spline_degree=2), tol=1e-9)
    ours = preset(n=16, spline_degree=2, tol=1e-9)
    assert ours.scheme == composed.scheme and isinstance(ours, iteration)
    a, b = ours.solve(problem), composed.solve(problem)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.info["history"] == b.info["history"]


@pytest.mark.parametrize(
    ("preset", "defaults"),
    [
        (FDNewton, {"n": 64, "max_iter": 50, "tol": 1e-10, "spline_degree": 3}),
        (FDPoisson, {"n": 64, "max_iter": 10_000, "tol": 1e-12, "spline_degree": 3}),
    ],
)
def test_preset_defaults_and_flat_hparams(preset, defaults):
    assert preset().hparams == {
        "class_path": f"maesolvers.{preset.__name__}",
        "init_args": defaults,
    }
    assert "scheme" not in repr(preset())


def test_composed_hparams_nest_the_scheme():
    config = PoissonIteration(scheme=Standard9(n=8), max_iter=3).hparams
    assert config == {
        "class_path": "maesolvers.PoissonIteration",
        "init_args": {
            "scheme": {
                "class_path": "maesolvers.Standard9",
                "init_args": {"n": 8, "spline_degree": 3},
            },
            "max_iter": 3,
            "tol": 1e-12,
        },
    }
    json.dumps(config)


def test_the_scheme_decides_what_is_supported():
    solver = Newton(scheme=Standard9(n=8))
    assert solver.supports(T1) and not solver.supports(T8)
    with pytest.raises(UnsupportedProblemError, match="Disk"):
        solver.solve(T8)


def test_an_iteration_needs_a_scheme_with_its_methods():
    with pytest.raises(
        TypeError,
        match="Newton needs a scheme with residual, jacobian, which PoissonOnly",
    ):
        Newton(scheme=PoissonOnly()).solve(T1)
    result = PoissonIteration(scheme=PoissonOnly()).solve(T1)
    assert result.info["converged"]


def test_the_scheme_must_be_a_frozen_dataclass_scheme_instance():
    class Plain(Scheme):
        """Not a dataclass; the four methods are stand-ins."""

        def supports(self, problem):
            return True

        discretize = rhs = solution = supports

    @dataclasses.dataclass
    class Mutable(Plain):
        pass

    @dataclasses.dataclass(frozen=True)
    class Frozen(Plain):
        pass

    class Undecorated(PoissonOnly):
        """Inherits the decoration, so a new field would miss from hparams."""

    with pytest.raises(TypeError, match="instance, got the class Standard9"):
        Newton(scheme=Standard9)
    with pytest.raises(
        TypeError, match=re.escape("PoissonIteration.scheme must subclass Scheme")
    ):
        PoissonIteration(scheme=object())
    for bad in (Plain(), Mutable(), Undecorated()):
        with pytest.raises(TypeError, match=f"frozen dataclass.*{type(bad).__name__}"):
            Newton(scheme=bad)
    assert Newton(scheme=Frozen()).scheme == Frozen()


def test_a_new_iteration_writes_three_hooks():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Jacobi(SchemeIteration):
        """Poisson iteration with the update damped by one half."""

        measure: ClassVar[str] = "update"
        requires: ClassVar[tuple[str, ...]] = ("second_derivatives", "solve_laplace")
        tol: float = 1e-10

        def initial_guess(self, problem, context):
            rhs = np.sqrt(2.0 * self.scheme.rhs(context))
            return self.scheme.solve_laplace(context, rhs)

        def step(self, problem, context, iterate):
            uxx, uyy, uxy = self.scheme.second_derivatives(context, iterate)
            lap = np.sqrt(uxx**2 + uyy**2 + 2 * uxy**2 + 2 * self.scheme.rhs(context))
            new = 0.5 * (iterate + self.scheme.solve_laplace(context, lap))
            return new, float(np.max(np.abs(new - iterate)))

        def converged(self, problem, context, measure):
            return measure <= self.tol

    solver = Jacobi(scheme=Standard9(n=16), max_iter=500)
    assert solver.supports(T1) and not solver.supports(T8)
    ours = solver.solve(T1)
    reference = PoissonIteration(scheme=Standard9(n=16)).solve(T1)
    assert ours.info["converged"]
    np.testing.assert_allclose(ours.values, reference.values, rtol=1e-8)


@pytest.mark.parametrize("iteration", [Newton, PoissonIteration])
def test_iterations_validate_tol_and_max_iter(iteration):
    with pytest.raises(ValueError, match="tol"):
        iteration(scheme=Standard9(), tol=-1.0)
    with pytest.raises(ValueError, match="max_iter"):
        iteration(scheme=Standard9(), max_iter=0)
    with pytest.raises(TypeError, match="scheme"):
        iteration()


def test_newton_and_poisson_reach_different_roots_on_t2():
    """Finding 2: same scheme, two iterations, two discrete solutions."""
    scheme = Standard9(n=32)
    newton = evaluate(Newton(scheme=scheme).solve(T2))
    poisson = evaluate(PoissonIteration(scheme=scheme).solve(T2))
    assert poisson["nonconvex_fraction"] == 0.0 < newton["nonconvex_fraction"]


def test_the_registry_lists_the_presets_read_only():
    assert dict(SOLVERS) == {"FDNewton": FDNewton, "FDPoisson": FDPoisson}
    with pytest.raises(TypeError):
        SOLVERS["Mine"] = FDNewton  # type: ignore[index]


@pytest.mark.parametrize("solver", list(SOLVERS.values()))
def test_registered_solvers_declare_their_support_on_the_class(solver):
    assert (solver.domains, solver.boundaries, solver.dims) == (
        Standard9.domains,
        Standard9.boundaries,
        Standard9.dims,
    )
    for problem in (T8, T1):
        assert solver().supports(problem) == Standard9().supports(problem)


@pytest.mark.parametrize("preset", [FDNewton, FDPoisson])
def test_preset_scheme_defaults_are_the_schemes(preset):
    assert preset().scheme == Standard9()


@pytest.mark.parametrize(
    "solver",
    [
        Newton(scheme=Standard9(n=8), max_iter=7),
        PoissonIteration(scheme=PoissonOnly(n=12), tol=1e-9),
        FDPoisson(n=10, spline_degree=2),
    ],
    ids=["Newton", "user scheme", "preset"],
)
def test_hparams_rebuild_the_solver_through_jsonargparse(solver):
    jsonargparse = pytest.importorskip("jsonargparse")
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(type(solver), "solver")
    config = parser.parse_object({"solver": solver.hparams["init_args"]})
    assert parser.instantiate(config).solver == solver


def test_a_scheme_is_set_from_the_command_line():
    jsonargparse = pytest.importorskip("jsonargparse")
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(Newton, "solver")
    args = ["--solver.scheme", "Standard9", "--solver.scheme.n", "16"]
    solver = parser.instantiate(parser.parse_args(args)).solver
    assert solver == Newton(scheme=Standard9(n=16))
