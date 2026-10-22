"""Iterations composed with a scheme, the presets, and the registry (ADR 0004)."""

import dataclasses
import json

import numpy as np
import pytest

from maesolvers import (
    P1,
    P4,
    P5,
    SOLVERS,
    FDNewton,
    FDPoisson,
    Newton,
    PoissonIteration,
    Standard9,
    UnsupportedProblemError,
)
from maesolvers.metrics import evaluate


@pytest.mark.parametrize(
    ("preset", "iteration"), [(FDNewton, Newton), (FDPoisson, PoissonIteration)]
)
@pytest.mark.parametrize("problem", [P4, P5])
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
        "class_path": f"maesolvers.classical.presets.{preset.__name__}",
        "init_args": defaults,
    }
    assert "scheme" not in repr(preset())


def test_composed_hparams_nest_the_scheme():
    config = PoissonIteration(scheme=Standard9(n=8), max_iter=3).hparams
    assert config == {
        "class_path": "maesolvers.classical.iterations.PoissonIteration",
        "init_args": {
            "scheme": {
                "class_path": "maesolvers.classical.schemes.Standard9",
                "init_args": {"n": 8, "spline_degree": 3},
            },
            "max_iter": 3,
            "tol": 1e-12,
        },
    }
    json.dumps(config)


def test_the_scheme_decides_what_is_supported():
    solver = Newton(scheme=Standard9(n=8))
    assert solver.supports(P4) and not solver.supports(P1)
    with pytest.raises(UnsupportedProblemError, match="Disk"):
        solver.solve(P1)


def test_an_iteration_needs_a_scheme_with_its_methods():
    @dataclasses.dataclass(frozen=True)
    class NoJacobian:
        inner: Standard9 = Standard9(n=8)

        def __getattr__(self, name):
            if name == "jacobian":
                raise AttributeError(name)
            return getattr(self.inner, name)

    with pytest.raises(TypeError, match="Newton needs a scheme with jacobian"):
        Newton(scheme=NoJacobian()).solve(P4)
    result = PoissonIteration(scheme=NoJacobian()).solve(P4)
    assert result.info["converged"]


def test_a_scheme_without_its_methods_is_rejected_before_setup():
    with pytest.raises(TypeError, match="discretize.*rhs"):
        PoissonIteration(scheme=object()).solve(P4)


@pytest.mark.parametrize("iteration", [Newton, PoissonIteration])
def test_iterations_validate_tol_and_max_iter(iteration):
    with pytest.raises(ValueError, match="tol"):
        iteration(scheme=Standard9(), tol=-1.0)
    with pytest.raises(ValueError, match="max_iter"):
        iteration(scheme=Standard9(), max_iter=0)
    with pytest.raises(TypeError, match="scheme"):
        iteration()


def test_newton_and_poisson_reach_different_roots_on_p5():
    """Finding 2: same scheme, two iterations, two discrete solutions."""
    scheme = Standard9(n=32)
    newton = evaluate(Newton(scheme=scheme).solve(P5))
    poisson = evaluate(PoissonIteration(scheme=scheme).solve(P5))
    assert poisson["nonconvex_fraction"] == 0.0 < newton["nonconvex_fraction"]


def test_the_registry_lists_the_presets_read_only():
    assert dict(SOLVERS) == {"FDNewton": FDNewton, "FDPoisson": FDPoisson}
    with pytest.raises(TypeError):
        SOLVERS["Mine"] = FDNewton  # type: ignore[index]
