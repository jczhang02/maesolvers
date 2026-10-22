"""The filtered scheme, its pure kernels and its preset."""

import dataclasses
import math

import numpy as np
import pytest
import scipy.sparse as sp

from maesolvers import (
    SOLVERS,
    T1,
    T2,
    T8,
    T20,
    Box,
    Dirichlet,
    FDFiltered,
    FDMonotone,
    Filtered,
    Monotone17,
    Newton,
    Problem,
    Scheme,
    Standard9,
    UnsupportedProblemError,
    from_config,
)
from maesolvers.classical.filtered import filter_function, filter_slope, filter_width
from maesolvers.metrics import evaluate

X0 = np.array([0.5, 0.5])


def _centred_exp(x):
    return np.exp(np.sum((np.asarray(x, dtype=float) - X0) ** 2, axis=-1) / 2)


def _centred_exp_f(x):
    r2 = np.sum((np.asarray(x, dtype=float) - X0) ** 2, axis=-1)
    return (1 + r2) * np.exp(r2)


CENTRED = Problem(
    "C2",
    Box((0.0, 0.0), (1.0, 1.0)),
    _centred_exp_f,
    Dirichlet(_centred_exp),
    exact=_centred_exp,
)
"""Froese-Oberman 2013, eq. (4.1): T1 centred at (1/2, 1/2)."""


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cells(Scheme):
    """A user scheme with no ``n``: Standard9 on ``cells`` cells per side."""

    cells: int = 8

    def _inner(self):
        return Standard9(n=self.cells)

    def supports(self, problem):
        return self._inner().supports(problem)

    def discretize(self, problem):
        return self._inner().discretize(problem)

    def rhs(self, grid):
        return self._inner().rhs(grid)

    def solution(self, grid, v):
        return self._inner().solution(grid, v)

    def residual(self, grid, v):
        return self._inner().residual(grid, v)

    def jacobian(self, grid, v):
        return self._inner().jacobian(grid, v)

    def solve_laplace(self, grid, rhs):
        return self._inner().solve_laplace(grid, rhs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoLaplace(Cells):
    """A monotone part with ``n`` and ``dtheta`` but no ``solve_laplace``."""

    n: int = 12
    angular_resolution: float = 0.1
    solve_laplace = None  # type: ignore[assignment]


def _filtered(n=12, stencil=17, epsilon=None):
    return Filtered(
        accurate=Standard9(n=n),
        monotone=Monotone17(n=n, stencil=stencil),
        epsilon=epsilon,
    )


def test_the_filter_is_the_identity_near_zero_and_zero_far_out():
    x = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    expected = [0.0, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
    np.testing.assert_array_equal(filter_function(x), expected)
    t = np.linspace(-4, 4, 801)
    assert np.all(np.abs(filter_function(t)) <= 1.0)
    np.testing.assert_array_equal(filter_function(-t), -filter_function(t))


def test_the_filter_slope_is_its_derivative_with_the_inner_side_at_the_kinks():
    x = np.array([-3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0])
    expected = [0.0, 0.0, -1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0]
    np.testing.assert_array_equal(filter_slope(x), expected)
    t = np.array([-2.7, -1.3, -0.4, 0.6, 1.8, 2.2])
    step = 1e-7
    numeric = (filter_function(t + step) - filter_function(t - step)) / (2 * step)
    np.testing.assert_allclose(filter_slope(t), numeric, atol=1e-6)


def test_the_filter_width_is_the_papers_rule():
    assert filter_width(1 / 30, math.pi / 8) == pytest.approx(
        math.sqrt(1 / 30) + math.pi / 80
    )


def test_the_default_width_reads_h_and_dtheta_from_the_monotone_part():
    grid = _filtered(n=16, stencil=33).discretize(T1)
    assert grid.width == pytest.approx(0.25 + math.atan(1 / 3) / 20)
    assert _filtered(epsilon=0.3).discretize(T1).width == 0.3


@pytest.mark.parametrize("epsilon", [0.05, 1e-3])
def test_the_residual_is_the_accurate_one_near_and_the_monotone_one_far(epsilon):
    scheme = _filtered(n=10, epsilon=epsilon)
    grid = scheme.discretize(T2)
    rng = np.random.default_rng(0)
    exact = T2.exact(grid.accurate.nodes[grid.accurate.interior])
    v = exact + 1e-3 * rng.standard_normal(exact.size)
    fa = Standard9(n=10).residual(grid.accurate, v)
    fm = Monotone17(n=10).residual(grid.monotone, v)
    r = scheme.residual(grid, v)
    x = (fa - fm) / epsilon
    near, far = np.abs(x) <= 1, np.abs(x) >= 2
    assert near.any() and far.any()
    np.testing.assert_allclose(r[near], fa[near], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(r[far], fm[far])
    np.testing.assert_allclose(r, fm + epsilon * filter_function(x), rtol=1e-14)
    assert np.all(np.abs(r - fm) <= epsilon * (1 + 1e-12))


def test_the_jacobian_is_the_papers_approximation():
    scheme = _filtered(n=10, epsilon=0.05)
    grid = scheme.discretize(T2)
    rng = np.random.default_rng(1)
    v = T2.exact(grid.accurate.nodes[grid.accurate.interior])
    v = v + 1e-3 * rng.standard_normal(v.size)
    ja = Standard9(n=10).jacobian(grid.accurate, v)
    jm = Monotone17(n=10).jacobian(grid.monotone, v)
    fa = Standard9(n=10).residual(grid.accurate, v)
    fm = Monotone17(n=10).residual(grid.monotone, v)
    s = filter_slope((fa - fm) / 0.05)
    assert set(s) == {-1.0, 0.0, 1.0}  # every piece of the filter is used
    expected = sp.diags(1 - s) @ jm + sp.diags(np.maximum(s, 0)) @ ja
    np.testing.assert_allclose(scheme.jacobian(grid, v).toarray(), expected.toarray())


def test_the_jacobian_is_the_accurate_one_on_a_smooth_problem():
    # Where the filter is the identity, the residual is the standard scheme's,
    # and so is its derivative.
    scheme = _filtered(n=8)
    grid = scheme.discretize(CENTRED)
    v = CENTRED.exact(grid.accurate.nodes[grid.accurate.interior])
    np.testing.assert_allclose(
        scheme.jacobian(grid, v).toarray(),
        Standard9(n=8).jacobian(grid.accurate, v).toarray(),
    )
    step = 1e-6
    e = np.zeros(v.size)
    e[20] = step
    numeric = (scheme.residual(grid, v + e) - scheme.residual(grid, v - e)) / (2 * step)
    np.testing.assert_allclose(
        scheme.jacobian(grid, v).toarray()[:, 20], numeric, atol=1e-5
    )


def test_on_the_smooth_centred_example_it_is_the_standard_scheme():
    """Froese-Oberman 2013, Table 4.1: the filtered columns equal the standard one."""
    for stencil in (9, 17, 33):
        result = FDFiltered(n=30, stencil=stencil).solve(CENTRED)
        assert result.info["converged"] and result.info["monotone_fraction"] == 0.0
        standard = Newton(scheme=Standard9(n=30)).solve(CENTRED)
        np.testing.assert_allclose(result.values, standard.values, atol=1e-12)


def test_far_from_the_accurate_scheme_it_is_the_monotone_scheme():
    result = FDFiltered(n=16, epsilon=1e-9).solve(T20)
    monotone = FDMonotone(n=16).solve(T20)
    assert result.info["converged"] and result.info["monotone_fraction"] > 0.5
    np.testing.assert_allclose(result.values, monotone.values, atol=1e-9)


def test_the_solution_is_the_accurate_parts_with_the_filter_diagnostics():
    result = FDFiltered(n=12, epsilon=0.2).solve(T20)
    assert result.info["interpolant"] == "spline(3)"
    assert result.info["filter_width"] == 0.2
    assert 0.0 <= result.info["monotone_fraction"] <= 1.0
    assert result.info["residual_max"] <= 1e-10
    assert result.info["unknowns"] == 121
    assert evaluate(result)["max"] < 2e-3


def test_newton_does_not_converge_on_t1_with_the_papers_jacobian():
    # A known limit (docs/methods/filtered.md): near T1's corner the filter
    # band holds the largest residual, where the paper's Jacobian (2 J_M for
    # 2 J_M - J_A) gives no direction of descent.
    result = FDFiltered(n=32, max_iter=20).solve(T1)
    assert not result.info["converged"] and result.info["residual_max"] > 1e-3
    assert evaluate(result)["max"] < 2e-4


def test_supported_problems_are_those_of_both_parts():
    scheme = _filtered()
    assert scheme.supports(T1) and not scheme.supports(T8)
    assert Standard9().supports(T8)
    with pytest.raises(UnsupportedProblemError, match="monotone scheme Monotone17"):
        scheme.discretize(T8)
    with pytest.raises(UnsupportedProblemError, match="FDFiltered does not support T8"):
        FDFiltered(n=4).solve(T8)
    oblong = Problem(
        "R", Box((0.0, 0.0), (2.0, 1.0)), _constant(1.0), Dirichlet(_constant(0.0))
    )
    with pytest.raises(UnsupportedProblemError, match=r"FDFiltered .* equal sides"):
        FDFiltered(n=4).solve(oblong)


def test_the_parts_must_share_their_unknowns():
    with pytest.raises(ValueError, match="one grid, got n = 8"):
        Filtered(accurate=Standard9(n=8), monotone=Monotone17(n=9))
    mixed = Filtered(accurate=Cells(cells=9), monotone=Monotone17(n=8))
    with pytest.raises(ValueError, match="Cells has 64 and Monotone17 49"):
        mixed.discretize(T1)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (dict(epsilon=0.0), ValueError, "epsilon must be positive"),
        (dict(accurate=Standard9), TypeError, "Filtered.accurate must be an instance"),
        (dict(monotone=NoLaplace()), TypeError, "monotone needs solve_laplace"),
        (dict(monotone=Cells(cells=12)), TypeError, "Cells has no n or angular"),
        (dict(accurate=T1), TypeError, "Filtered.accurate must subclass Scheme"),
    ],
)
def test_filtered_validates_its_parts(kwargs, error, message):
    parts = dict(accurate=Standard9(n=12), monotone=Monotone17(n=12)) | kwargs
    with pytest.raises(error, match=message):
        Filtered(**parts)


def test_an_explicit_width_lifts_the_need_for_n():
    scheme = Filtered(accurate=Cells(cells=8), monotone=Cells(cells=8), epsilon=0.1)
    assert Newton(scheme=scheme).solve(T1).info["converged"]


def test_filtered_is_a_frozen_keyword_only_part():
    assert _filtered(epsilon=np.float64(0.5)) == _filtered(epsilon=0.5)
    assert type(_filtered(epsilon=np.float64(0.5)).epsilon) is float
    with pytest.raises(TypeError):
        Filtered(Standard9(), Monotone17())  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _filtered().epsilon = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("problem", [CENTRED, T20])
def test_the_preset_is_its_composition(problem):
    composed = Newton(scheme=_filtered(n=12, stencil=33), max_iter=200)
    ours = FDFiltered(n=12, stencil=33)
    assert ours.scheme == composed.scheme and isinstance(ours, Newton)
    a, b = ours.solve(problem), composed.solve(problem)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.info["history"] == b.info["history"]


def test_the_preset_has_flat_hparams_and_its_parts_defaults():
    assert FDFiltered().hparams == {
        "class_path": "maesolvers.FDFiltered",
        "init_args": {
            "n": 64,
            "stencil": 17,
            "boundary": "narrow",
            "jacobian_floor": 1e-12,
            "epsilon": None,
            "spline_degree": 3,
            "max_iter": 200,
            "tol": 1e-10,
        },
    }
    assert FDFiltered().scheme == Filtered(accurate=Standard9(), monotone=Monotone17())
    assert SOLVERS["FDFiltered"] is FDFiltered
    assert (FDFiltered.domains, FDFiltered.boundaries, FDFiltered.dims) == (
        Monotone17.domains,
        Monotone17.boundaries,
        Monotone17.dims,
    )


@pytest.mark.parametrize(
    "solver",
    [
        FDFiltered(n=10, stencil=33, boundary="interpolate", epsilon=0.25),
        Newton(scheme=_filtered(n=10, stencil=9), max_iter=7),
    ],
    ids=["preset", "composed"],
)
def test_hparams_rebuild_the_solver_through_jsonargparse(solver):
    jsonargparse = pytest.importorskip("jsonargparse")
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(type(solver), "solver")
    config = parser.parse_object({"solver": solver.hparams["init_args"]})
    assert parser.instantiate(config).solver == solver


def test_a_composed_filtered_solver_comes_from_a_config():
    pytest.importorskip("jsonargparse")
    config = {
        "class_path": "maesolvers.Newton",
        "init_args": {
            "scheme": {
                "class_path": "maesolvers.Filtered",
                "init_args": {
                    "accurate": {"class_path": "Standard9", "init_args": {"n": 8}},
                    "monotone": {
                        "class_path": "Monotone17",
                        "init_args": {"n": 8, "stencil": 9},
                    },
                },
            }
        },
    }
    assert from_config(config) == Newton(scheme=_filtered(n=8, stencil=9))
