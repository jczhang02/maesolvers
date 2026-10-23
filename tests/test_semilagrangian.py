"""The semi-Lagrangian scheme of Feng-Jensen 2017, Howard's algorithm and the preset."""

import dataclasses
import itertools

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import Delaunay

from maesolvers import (
    T1,
    T8,
    T9,
    T12,
    T20,
    T39,
    T40,
    Box,
    Dirichlet,
    FDNewton,
    FDSemiLagrangian,
    Howard,
    Newton,
    Problem,
    SemiLagrangian,
    Standard9,
    UnsupportedProblemError,
    from_config,
)
from maesolvers.classical import _linear
from maesolvers.classical.semilagrangian import (
    fit_hessian,
    frame_values,
    frame_weights,
    linear_weights,
    semi_lagrangian_operators,
    stencil_directions,
)
from maesolvers.metrics import evaluate

UNIT = Box((0.0, 0.0), (1.0, 1.0))
HESSIAN = np.array([[2.0, 1.0], [1.0, 6.0]])


def _quadratic(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.einsum("...i,ij,...j->...", x, HESSIAN, x) + x[..., 0] - x[..., 1]


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


QUADRATIC = Problem(
    "quadratic", UNIT, _constant(np.linalg.det(HESSIAN)), Dirichlet(_quadratic)
)


def _steps(scheme, problem, count):
    """The first ``count`` iterates of Howard's algorithm, the start first."""
    howard = Howard(scheme=scheme)
    grid = scheme.discretize(problem)
    iterates = [howard.initial_guess(problem, grid)]
    for _ in range(count):
        iterates.append(howard.step(problem, grid, iterates[-1])[0])
    return grid, iterates


# Kernels


def test_stencil_directions_form_orthogonal_frames():
    s = stencil_directions(8)
    assert s.shape == (8, 2)
    np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0)
    np.testing.assert_allclose(s[0], [1.0, 0.0])
    np.testing.assert_allclose(s[4], [0.0, 1.0], atol=1e-16)
    np.testing.assert_allclose(np.sum(s[:4] * s[4:], axis=1), 0.0, atol=1e-16)
    angles = np.arctan2(s[:, 1], s[:, 0])
    np.testing.assert_allclose(np.diff(angles), np.pi / 8)


@pytest.mark.parametrize("count", [3, 0, 2.5])
def test_stencil_directions_must_be_an_even_count(count):
    with pytest.raises((ValueError, TypeError), match="directions"):
        stencil_directions(count)


def test_frame_values_are_the_supremum_over_the_split():
    rng = np.random.default_rng(0)
    d1, d2 = 3.0 * rng.normal(size=(2, 200))
    f = rng.uniform(0.0, 5.0, size=200)
    a = np.linspace(0.0, 1.0, 20001)[:, None]
    brute = np.max(-(a * d1 + (1 - a) * d2) + 2 * np.sqrt(f * a * (1 - a)), axis=0)
    np.testing.assert_allclose(frame_values(d1, d2, f), brute, atol=1e-6)


def test_frame_values_vanish_on_a_convex_pair_with_the_right_determinant():
    d1, d2 = np.array([2.0, 4.0, 0.0, -1.0]), np.array([3.0, 0.5, 5.0, -2.0])
    f = np.array([6.0, 2.0, 0.0, 2.0])
    values = frame_values(d1, d2, f)
    np.testing.assert_allclose(values[:3], 0.0, atol=1e-15)
    assert values[3] > 0  # d1 d2 = f, but both second differences negative


def test_frame_values_are_nonincreasing_in_both_differences():
    rng = np.random.default_rng(1)
    d1, d2 = rng.normal(size=(2, 500))
    f = rng.uniform(0.0, 2.0, size=500)
    base = frame_values(d1, d2, f)
    assert np.all(frame_values(d1 + 0.1, d2, f) <= base)
    assert np.all(frame_values(d1, d2 + 0.1, f) <= base)


def test_frame_weights_are_minus_the_derivatives_of_frame_values():
    rng = np.random.default_rng(2)
    d1, d2 = rng.normal(size=(2, 100))
    f = rng.uniform(0.1, 2.0, size=100)
    a, eps = frame_weights(d1, d2, f), 1e-7
    by_d1 = (frame_values(d1 + eps, d2, f) - frame_values(d1 - eps, d2, f)) / (2 * eps)
    by_d2 = (frame_values(d1, d2 + eps, f) - frame_values(d1, d2 - eps, f)) / (2 * eps)
    np.testing.assert_allclose(-by_d1, a, atol=1e-7)
    np.testing.assert_allclose(-by_d2, 1.0 - a, atol=1e-7)
    assert np.all((a >= 0.0) & (a <= 1.0))


def test_frame_weights_split_evenly_where_every_split_attains_the_supremum():
    assert frame_weights(np.array([1.5]), np.array([1.5]), np.array([0.0])) == 0.5
    # f = 0: the weight goes to the smaller difference.
    np.testing.assert_allclose(
        frame_weights(np.array([1.0, 3.0]), np.array([3.0, 1.0]), 0.0), [1.0, 0.0]
    )


def test_fit_hessian_recovers_a_quadratic():
    s = stencil_directions(6)
    second = np.einsum("di,ij,dj->d", s, HESSIAN, s)[:, None] * [1.0, 2.0]
    fitted = fit_hessian(s, second)
    np.testing.assert_allclose(fitted, [HESSIAN, 2 * HESSIAN], atol=1e-13)


def test_linear_weights_reproduce_linear_functions_with_nonnegative_weights():
    rng = np.random.default_rng(3)
    tri = Delaunay(rng.uniform(size=(40, 2)))
    inside = tri.points[tri.simplices].mean(axis=1)  # one point per triangle
    vertices, weights = linear_weights(tri, inside)
    linear = tri.points @ [2.0, -3.0] + 0.5
    np.testing.assert_allclose(
        np.sum(weights * linear[vertices], axis=1), inside @ [2.0, -3.0] + 0.5
    )
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="outside the triangulation"):
        linear_weights(tri, np.array([[5.0, 5.0]]))


def test_operators_vanish_on_linear_functions_and_have_the_m_matrix_signs():
    tri = Delaunay(np.random.default_rng(4).uniform(size=(80, 2)))
    unknown = np.flatnonzero(np.all((tri.points > 0.2) & (tri.points < 0.8), axis=1))
    reach = np.full(unknown.size, 0.1)
    s = stencil_directions(4)
    ops = semi_lagrangian_operators(tri, unknown, s, reach)
    assert ops.shape == (4 * unknown.size, len(tri.points))
    linear = tri.points @ [1.0, 2.0] - 3.0
    np.testing.assert_allclose(ops @ linear, 0.0, atol=1e-11)
    centre = ops[:, unknown][
        np.arange(4 * unknown.size), np.tile(np.arange(unknown.size), 4)
    ]
    off = ops.copy().tolil()
    off[np.arange(4 * unknown.size), np.tile(unknown, 4)] = 0.0
    # Each row: -2 / k^2 at the centre, up to the interpolation weight of the
    # centre itself, and nonnegative weights elsewhere.
    assert np.all(np.asarray(centre).ravel() < 0.0)
    assert sp.csr_matrix(off).min() >= 0.0


# The scheme


@pytest.mark.parametrize("problem", [QUADRATIC, T1, T39, T8])
def test_the_stencil_stays_in_the_triangulation_and_is_m_h_inside(problem):
    scheme = SemiLagrangian(n=16, stencil_factor=2.0)
    grid = scheme.discretize(problem)
    h = max(grid.h)
    assert np.all(grid.reach <= 2.0 * h + 1e-15) and np.all(grid.reach > 0.0)
    # Far from the boundary the stencil has its full size.
    box = problem.domain.bbox
    centre = np.mean([box.lo, box.hi], axis=0)
    near_centre = np.linalg.norm(grid.nodes[grid.interior] - centre, axis=1) < 0.1
    np.testing.assert_allclose(grid.reach[near_centre], 2.0 * h)


def test_second_differences_of_a_quadratic_are_accurate_where_the_stencil_is_full():
    scheme = SemiLagrangian(n=32, stencil_factor=4.0)
    grid = scheme.discretize(QUADRATIC)
    d = scheme.second_differences(grid, _quadratic(grid.nodes[grid.interior]))
    exact = np.einsum("di,ij,dj->d", grid.directions, HESSIAN, grid.directions)
    full = np.isclose(grid.reach, 4.0 * max(grid.h))
    # The interpolation error of a quadratic is at most |H| h^2 / 4 at each
    # end; divided by k^2 / 2 it is at most |H| / (2 m^2).
    bound = np.linalg.norm(HESSIAN, 2) / (2 * 4.0**2)
    assert np.max(np.abs(d[:, full] - exact[:, None])) <= bound


def test_jacobian_matches_finite_differences_of_the_residual():
    scheme = SemiLagrangian(n=8, stencil_factor=2.0, directions=8)
    grid = scheme.discretize(T39)
    rng = np.random.default_rng(5)
    v = T39.exact(grid.nodes[grid.interior]) + 0.01 * rng.normal(size=grid.f.size)
    jac = scheme.jacobian(grid, v).toarray()
    eps = 1e-7
    for j in rng.choice(v.size, size=10, replace=False):
        e = np.zeros_like(v)
        e[j] = eps
        column = (scheme.residual(grid, v + e) - scheme.residual(grid, v - e)) / (
            2 * eps
        )
        np.testing.assert_allclose(column, jac[:, j], atol=1e-4)


def test_the_jacobian_is_an_m_matrix():
    """Feng-Jensen 2017, Lemma 5.1, at any iterate, convex or not."""
    scheme = SemiLagrangian(n=12, stencil_factor=3.0)
    grid = scheme.discretize(T39)
    v = np.random.default_rng(6).normal(size=grid.f.size)
    jac = scheme.jacobian(grid, v).toarray()
    off = jac - np.diag(np.diag(jac))
    assert np.all(np.diag(jac) > 0.0) and np.all(off <= 0.0)
    assert np.all(np.diag(jac) + off.sum(axis=1) >= -1e-9 * np.diag(jac))


def test_solve_laplace_is_the_control_one_half_identity():
    """``d_x + d_y = 2 sqrt(f)`` is the equation of B = Id / 2 (Algorithm 1's start)."""
    scheme = SemiLagrangian(n=12)
    grid = scheme.discretize(T39)
    v = scheme.solve_laplace(grid, 2.0 * np.sqrt(grid.f))
    d = scheme.second_differences(grid, v)
    half = len(grid.directions) // 2
    a = 0.5
    control = -(a * d[0] + (1 - a) * d[half]) + 2 * np.sqrt(grid.f * a * (1 - a))
    np.testing.assert_allclose(control, 0.0, atol=1e-8 * np.max(grid.f))


def test_a_negative_or_infinite_f_is_an_error_naming_the_problem():
    for value in (-1.0, np.inf):
        problem = Problem("bad", UNIT, _constant(value), Dirichlet(_quadratic))
        with pytest.raises(ValueError, match="bad"):
            SemiLagrangian(n=4).discretize(problem)


@pytest.mark.parametrize(
    ("problem", "supported"),
    [(T1, True), (T8, True), (T39, True), (T40, True), (T12, False), (T9, False)],
)
def test_supports_convex_2d_dirichlet_problems(problem, supported):
    assert SemiLagrangian().supports(problem) == supported
    assert FDSemiLagrangian().supports(problem) == supported
    if not supported:
        with pytest.raises(UnsupportedProblemError, match="SemiLagrangian"):
            SemiLagrangian(n=4).discretize(problem)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n": 1}, ValueError, "n"),
        ({"stencil_factor": 0.0}, ValueError, "stencil_factor"),
        ({"directions": 7}, ValueError, "even"),
        ({"directions": 0}, ValueError, "directions"),
    ],
)
def test_semi_lagrangian_validates_its_parameters(kwargs, error, message):
    with pytest.raises(error, match=message):
        SemiLagrangian(**kwargs)


def test_semi_lagrangian_is_a_frozen_keyword_only_part():
    scheme = SemiLagrangian(n=8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scheme.n = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        SemiLagrangian(8)  # type: ignore[misc]


# Howard's algorithm


def test_howard_decreases_monotonically_after_the_first_policy():
    """Feng-Jensen 2017, Theorem 5.2: the iterates decrease monotonically."""
    _, iterates = _steps(SemiLagrangian(n=16), T39, 5)
    for before, after in itertools.pairwise(iterates[1:]):
        assert np.all(after <= before + 1e-12)


def test_a_howard_step_solves_the_maximising_controls_equation():
    scheme = SemiLagrangian(n=12)
    grid, (v0, v1) = _steps(scheme, T39, 1)
    # Newton with the Danskin Jacobian: the residual is affine in v for the
    # control chosen at v0, so the step is exact for that control.
    jac = scheme.jacobian(grid, v0)
    affine = scheme.residual(grid, v0) + jac @ (v1 - v0)
    np.testing.assert_allclose(affine, 0.0, atol=1e-8)


@pytest.mark.parametrize("problem", [T39, T40, T20])
def test_the_root_is_convex_along_every_direction(problem):
    scheme = SemiLagrangian(n=15, stencil_factor=2.0)
    result = Howard(scheme=scheme, tol=1e-12).solve(problem)
    assert result.info["converged"]
    grid = scheme.discretize(problem)
    v = result.values[result.interior]
    d = scheme.second_differences(grid, v)
    assert d.min() >= -1e-9 * np.abs(d).max()
    assert np.max(np.abs(scheme.residual(grid, v))) <= 1e-9 * np.abs(d).max()


def test_newton_and_howard_reach_the_same_root():
    """The discrete solution is unique (Feng-Jensen 2017, Theorem 5.2)."""
    scheme = SemiLagrangian(n=16)
    howard = Howard(scheme=scheme, tol=1e-12).solve(T39)
    newton = Newton(scheme=scheme, tol=1e-12).solve(T39)
    np.testing.assert_allclose(howard.values, newton.values, atol=1e-9)


def test_howard_on_the_standard_scheme_is_undamped_newton():
    howard = Howard(scheme=Standard9(n=16), tol=1e-12).solve(T1)
    newton = FDNewton(n=16).solve(T1)
    np.testing.assert_allclose(howard.values, newton.values, atol=1e-10)


def test_howard_converges_in_a_few_steps_and_the_error_falls_with_h():
    """Fig. 8 of the paper: 5 to 16 steps."""
    errors = []
    for n in (16, 32):
        result = FDSemiLagrangian(n=n, stencil_factor=2.0).solve(T39)
        assert result.info["converged"] and result.info["iterations"] <= 16
        errors.append(evaluate(result)["max"])
    assert errors[1] < 0.7 * errors[0]


@pytest.mark.parametrize("n", [15, 16])
def test_t40_is_exact_at_the_nodes_because_its_kink_follows_the_grid(n):
    """benchmarks/feng_jensen2017.md, Finding 1: the kink x1 = 0 runs along
    the grid columns and the stencil direction (0, 1), so the interpolant of
    |x1| solves the scheme; turned off the grid, the kink costs accuracy."""
    assert evaluate(FDSemiLagrangian(n=n).solve(T40))["max"] < 1e-12
    nu = np.array([np.cos(0.3), np.sin(0.3)])
    turned = Problem(
        "turned T40",
        T40.domain,
        T40.f,
        Dirichlet(lambda x: np.abs(np.asarray(x) @ nu)),
        exact=lambda x: np.abs(np.asarray(x) @ nu),
    )
    assert evaluate(FDSemiLagrangian(n=n).solve(turned))["max"] > 1e-2


# The solution


def test_solution_interpolates_linearly_and_extends_outside_the_triangulation():
    result = FDSemiLagrangian(n=8).solve(T8)
    np.testing.assert_allclose(result.u(result.nodes), result.values, atol=1e-14)
    # A point of the disk outside the polygon of the boundary points takes
    # the value at the nearest point of the polygon, which is the boundary
    # value 0 there up to the interpolation along an edge.
    sliver = np.array([np.cos(0.1), np.sin(0.1)]) * (1 - 1e-9)
    assert abs(result.u(sliver)) < 1e-6
    assert result.info["interpolant"] == "linear"
    assert set(result.info) >= {"h", "residual_max", "min_reach", "interpolant"}


def test_solution_hessian_is_the_fit_at_the_unknowns_and_nan_at_the_boundary():
    scheme = SemiLagrangian(n=8)
    result = Howard(scheme=scheme).solve(QUADRATIC)
    hess = result.node_hessian
    assert np.all(np.isnan(hess[~result.interior]))
    assert np.all(np.isfinite(hess[result.interior]))


# The preset


def test_the_preset_is_its_composition():
    composed = Howard(scheme=SemiLagrangian(n=12, stencil_factor=3.0, directions=16))
    ours = FDSemiLagrangian(n=12, stencil_factor=3.0, directions=16)
    assert ours.scheme == composed.scheme and isinstance(ours, Howard)
    a, b = ours.solve(T39), composed.solve(T39)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.info["history"] == b.info["history"]


def test_the_preset_has_flat_hparams():
    assert FDSemiLagrangian().hparams == {
        "class_path": "maesolvers.FDSemiLagrangian",
        "init_args": {
            "max_iter": 50,
            "tol": 1e-6,
            "n": 64,
            "stencil_factor": 4.0,
            "directions": 64,
        },
    }


def test_the_composition_rebuilds_from_its_config():
    pytest.importorskip("jsonargparse")
    solver = Howard(scheme=SemiLagrangian(n=10, stencil_factor=2.5), tol=1e-8)
    assert from_config(solver.hparams) == solver
    preset = FDSemiLagrangian(n=10, directions=32)
    assert from_config(preset.hparams) == preset


# The linear solves


def _m_matrix(size, seed):
    rng = np.random.default_rng(seed)
    off = -sp.random(size, size, density=0.05, random_state=rng)
    off.setdiag(0.0)
    diagonal = 1.0 - np.asarray(off.sum(axis=1)).ravel()
    return sp.csr_matrix(off + sp.diags(diagonal)), rng.normal(size=size)


def test_solve_m_matrix_matches_a_direct_solve():
    matrix, rhs = _m_matrix(300, 7)
    expected = sp.linalg.spsolve(matrix.tocsc(), rhs)
    ours = _linear.solve_m_matrix(matrix, rhs)
    np.testing.assert_allclose(ours, expected, atol=1e-8 * np.abs(expected).max())
    np.testing.assert_array_equal(_linear.solve_m_matrix(matrix, 0 * rhs), 0.0)


@pytest.mark.parametrize("failing", [("bicgstab",), ("bicgstab", "gmres")])
def test_solve_m_matrix_falls_back_when_a_krylov_method_fails(monkeypatch, failing):
    for name in failing:
        monkeypatch.setattr(
            _linear, name, lambda a, b, **_: (np.full_like(b, np.nan), -10)
        )
    matrix, rhs = _m_matrix(200, 8)
    expected = sp.linalg.spsolve(matrix.tocsc(), rhs)
    ours = _linear.solve_m_matrix(matrix, rhs)
    np.testing.assert_allclose(ours, expected, atol=1e-8 * np.abs(expected).max())
