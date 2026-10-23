"""The monotone mixed scheme of Chen-Wan-Lin 2018, its kernels, policy iteration and the preset."""

import numpy as np
import pytest
import scipy.sparse as sp

from maesolvers import (
    T1,
    T2,
    T8,
    Box,
    Dirichlet,
    FDMixed,
    MixedHJB,
    Newton,
    PolicyIteration,
    Problem,
    Standard9,
    UnsupportedProblemError,
    from_config,
)
from maesolvers.classical.mixed import (
    _directions,
    _wide_differences,
    control_coefficients,
    seven_point_band,
    seven_point_case,
    seven_point_controls,
    wide_stencil_value,
    wide_stencil_weight,
)
from maesolvers.metrics import evaluate

SQUARE = Box((-0.5, -0.5), (0.5, 0.5))


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


def _quadratic_problem(hessian, domain=SQUARE):
    """``u = x^T H x / 2 + x``, ``f = det H``, Dirichlet data from ``u``."""
    hessian = np.asarray(hessian, dtype=float)

    def u(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * np.einsum("...i,ij,...j->...", x, hessian, x) + x[..., 0]

    det = float(np.linalg.det(hessian))
    return Problem("Q", domain, _constant(det), Dirichlet(u), exact=u)


def _control_grid(count=801):
    """A dense grid over the control set ``[0, 1] x [-pi/4, pi/4]``, flattened."""
    a, theta = np.meshgrid(
        np.linspace(0.0, 1.0, count),
        np.linspace(-np.pi / 4, np.pi / 4, count),
        indexing="ij",
    )
    return a.ravel(), theta.ravel()


# The pure kernels


def test_control_coefficients_are_a_trace_one_matrix_with_eigenvalues_a():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, 50)
    theta = rng.uniform(-np.pi / 4, np.pi / 4, 50)
    a11, a22, a12 = control_coefficients(a, theta)
    matrices = np.stack([np.stack([a11, a12], -1), np.stack([a12, a22], -1)], -2)
    eig = np.linalg.eigvalsh(matrices)
    assert np.allclose(eig, np.sort(np.stack([a, 1.0 - a], -1), axis=-1))
    # e_z = (cos theta, -sin theta) is the eigenvector of a.
    e_z = np.stack([np.cos(theta), -np.sin(theta)], -1)
    assert np.allclose(np.einsum("kij,kj->ki", matrices, e_z), a[:, None] * e_z)


def test_the_band_is_where_a_seven_point_stencil_is_monotone():
    a, theta = _control_grid(301)
    low, high = seven_point_band(theta)
    inside = (a >= low - 1e-12) & (a <= high + 1e-12)
    outside = (a <= low - 1e-9) | (a >= high + 1e-9)
    case = seven_point_case(a, theta)
    assert np.all(case[outside] == 0)
    strictly = (a >= low + 1e-9) & (a <= high - 1e-9)
    assert np.all(case[strictly] > 0) and np.all(inside[case > 0])
    # Case 1 is a12 >= 0: a <= 1/2 for theta >= 0, a >= 1/2 for theta <= 0.
    _, _, a12 = control_coefficients(a, theta)
    assert np.all(a12[case == 1] >= 0.0) and np.all(a12[case == 2] < 0.0)
    assert seven_point_band(np.array(0.0)) == (0.0, 1.0)
    assert np.allclose(seven_point_band(np.array(np.pi / 4)), (0.0, 1.0))
    assert np.allclose(
        seven_point_band(np.array(np.pi / 8)), 0.5 + np.array([-1, 1]) / 8**0.5
    )


@pytest.mark.parametrize("zero_f", [False, True], ids=["f > 0", "f = 0"])
def test_seven_point_controls_match_a_brute_force_search(zero_f):
    rng = np.random.default_rng(1)
    k = 60
    dxx, dyy, dxy1, dxy2 = rng.normal(size=(4, k))
    f = np.zeros(k) if zero_f else rng.uniform(0.0, 2.0, k)
    best = seven_point_controls(dxx, dyy, dxy1, dxy2, f)
    a, theta = _control_grid()
    case = seven_point_case(a, theta)
    a11, a22, a12 = control_coefficients(a, theta)
    for p in range(k):
        dxy = np.where(case == 1, dxy1[p], dxy2[p])
        values = -a11 * dxx[p] - 2 * a12 * dxy - a22 * dyy[p]
        values += 2 * np.sqrt(np.maximum(a * (1 - a), 0) * f[p])
        brute = values[case > 0].max()
        # never below the search, which a closed form on a 801^2 grid beats
        assert best.value[p] >= brute - 1e-12
        assert best.value[p] <= brute + 1e-2 * (1 + abs(brute))
    # The value is the objective at the control, with its own stencil.
    a11, a22, a12 = control_coefficients(best.a, best.theta)
    dxy = np.where(best.case == 1, dxy1, dxy2)
    value = -a11 * dxx - 2 * a12 * dxy - a22 * dyy
    value += 2 * np.sqrt(np.maximum(best.a * (1 - best.a), 0) * f)
    assert np.allclose(best.value, value)
    assert set(np.unique(best.case)) <= {1, 2}


def test_on_a_convex_hessian_the_seven_point_maximum_is_zero_where_det_is_f():
    # Diagonally dominant: the optimal control A ~ H^-1 is a 7-point one.
    hessian = np.array([[2.0, 0.7], [0.7, 1.0]])
    f = np.array([np.linalg.det(hessian)])
    best = seven_point_controls(
        np.array([2.0]), np.array([1.0]), np.array([0.7]), np.array([0.7]), f
    )
    assert best.value[0] == pytest.approx(0.0, abs=1e-14)
    a11, a22, a12 = control_coefficients(best.a, best.theta)
    control = np.array([[a11[0], a12[0]], [a12[0], a22[0]]])
    inverse = np.linalg.inv(hessian)
    assert np.allclose(control, inverse / np.trace(inverse))


@pytest.mark.parametrize("restricted", [True, False])
def test_the_wide_stencil_weight_maximises_over_its_controls(restricted):
    rng = np.random.default_rng(2)
    dzz, dww = rng.normal(size=(2, 40))
    f = np.concatenate([np.zeros(10), rng.uniform(0.0, 2.0, 30)])
    theta = rng.uniform(-np.pi / 4, np.pi / 4, 40)
    a = wide_stencil_weight(dzz, dww, f, theta, restricted)
    candidates = np.linspace(0.0, 1.0, 20001)
    for p in range(40):
        allowed = candidates
        if restricted:
            low, high = seven_point_band(theta[p])
            allowed = candidates[(candidates <= low) | (candidates >= high)]
            assert a[p] <= low + 1e-12 or a[p] >= high - 1e-12
        values = wide_stencil_value(dzz[p], dww[p], allowed, f[p])
        assert wide_stencil_value(dzz[p], dww[p], a[p], f[p]) >= values.max() - 1e-12


# The discretisation


def test_the_wide_stencil_is_exact_on_a_bilinear_function_up_to_the_boundary():
    """u = x y: bilinear interpolation is exact, and along a line u is a
    quadratic with second derivative 2 e_x e_y, which the three-point
    difference with shortened arms and the boundary data reproduces."""
    problem = Problem(
        "xy", SQUARE, _constant(0.0), Dirichlet(lambda x: x[..., 0] * x[..., 1])
    )
    scheme = MixedHJB(n=12, angles=5)
    grid = scheme.discretize(problem)
    v = problem.boundary.g(grid.nodes[grid.interior])
    u = np.empty(len(grid.nodes))
    u[grid.interior], u[~grid.interior] = v, grid.boundary_values
    for theta in grid.angles:
        dzz, dww = _wide_differences(grid, grid.index, np.asarray(theta), u, v)
        e_z, e_w = _directions(np.asarray(theta))
        assert np.allclose(dzz, 2 * e_z[0] * e_z[1], atol=1e-10)
        assert np.allclose(dww, 2 * e_w[0] * e_w[1], atol=1e-10)


@pytest.mark.parametrize("seven_point", [True, False])
def test_every_policy_matrix_is_an_m_matrix(seven_point):
    scheme = MixedHJB(n=10, seven_point=seven_point)
    problem = _quadratic_problem([[1.0, 1.5], [1.5, 4.0]])
    grid = scheme.discretize(problem)
    rng = np.random.default_rng(3)
    for v in (rng.normal(size=81), problem.exact(grid.nodes[grid.interior])):
        matrix, rhs = scheme.policy_system(grid, v)
        dense = matrix.toarray()
        off = dense - np.diag(np.diag(dense))
        assert np.all(np.diag(dense) > 0) and np.all(off <= 1e-12)
        assert np.all(np.diag(dense) >= -off.sum(axis=1) - 1e-9)
        assert np.all(np.linalg.inv(dense) >= -1e-12)
        # A v - F is the maximum over the controls, the residual.
        assert np.allclose(matrix @ v - rhs, scheme.residual(grid, v))


def test_a_dominant_quadratic_is_a_root_with_seven_point_controls_only():
    problem = _quadratic_problem([[2.0, 0.7], [0.7, 1.0]])
    result = FDMixed(n=16).solve(problem)
    assert result.info["converged"] and result.info["wide_fraction"] == 0.0
    assert evaluate(result)["max"] < 1e-12


def test_a_quadratic_off_the_seven_point_controls_takes_the_wide_stencil():
    problem = _quadratic_problem([[1.0, 1.5], [1.5, 4.0]])
    mixed = FDMixed(n=16).solve(problem)
    assert mixed.info["converged"] and mixed.info["wide_fraction"] > 0.1
    assert 0.0 < evaluate(mixed)["max"] < 0.05


def test_policy_iteration_converges_at_second_order_on_t1():
    """Chen-Wan-Lin 2018, Example 1: 7-point controls at every node."""
    errors = []
    for n in (9, 17, 33):
        result = FDMixed(n=n).solve(T1)
        assert result.info["converged"] and result.info["iterations"] <= 6
        assert result.info["wide_fraction"] == 0.0
        assert result.info["residual_max"] <= 1e-6
        errors.append(evaluate(result)["max"])
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8)


def test_the_pure_wide_stencil_scheme_is_less_accurate_on_t1():
    mixed = evaluate(FDMixed(n=17).solve(T1))["max"]
    wide = PolicyIteration(scheme=MixedHJB(n=17, seven_point=False)).solve(T1)
    assert wide.info["converged"] and wide.info["wide_fraction"] == 1.0
    assert evaluate(wide)["max"] > 3 * mixed


def test_newton_on_the_scheme_reaches_the_policy_iteration_root():
    scheme = MixedHJB(n=12)
    howard = PolicyIteration(scheme=scheme).solve(T2)
    newton = Newton(scheme=scheme, tol=1e-12).solve(T2)
    assert newton.info["converged"]
    assert np.max(np.abs(newton.values - howard.values)) < 1e-9


def test_the_history_holds_the_residual_before_each_update():
    result = FDMixed(n=12).solve(T1)
    history = result.info["history"]
    assert len(history) == result.info["iterations"]
    assert history[-1] <= 1e-6 < history[-2]
    assert result.info["residual_max"] == pytest.approx(history[-1], abs=1e-12)


# Support and parameters


def test_the_scheme_needs_a_box_with_equal_sides_and_dirichlet_data():
    oblong = Problem(
        "R", Box((0.0, 0.0), (1.0, 2.0)), _constant(1.0), Dirichlet(_constant(0.0))
    )
    scheme = MixedHJB(n=8)
    assert scheme.supports(T1) and not scheme.supports(oblong)
    assert not scheme.supports(T8)
    with pytest.raises(UnsupportedProblemError, match="equal sides"):
        FDMixed(n=8).solve(oblong)
    with pytest.raises(UnsupportedProblemError):
        FDMixed(n=8).solve(T8)


def test_a_negative_f_is_refused():
    problem = Problem("neg", SQUARE, _constant(-1.0), Dirichlet(_constant(0.0)))
    with pytest.raises(ValueError, match="f >= 0"):
        MixedHJB(n=8).discretize(problem)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n": 1}, ValueError),
        ({"angles": 0}, ValueError),
        ({"angles": 2.5}, TypeError),
        ({"seven_point": "yes"}, TypeError),
        ({"spline_degree": 6}, ValueError),
    ],
)
def test_parameters_are_checked(kwargs, error):
    with pytest.raises(error):
        MixedHJB(**kwargs)


def test_angles_default_to_the_interior_nodes_per_side():
    grid = MixedHJB(n=9).discretize(T1)
    assert len(grid.angles) == 8 and grid.angles[0] == -np.pi / 4
    assert np.allclose(np.diff(grid.angles), np.pi / 16)
    assert len(MixedHJB(n=9, angles=3).discretize(T1).angles) == 3


def test_policy_iteration_refuses_a_scheme_without_a_policy_system():
    with pytest.raises(TypeError, match="policy_system"):
        PolicyIteration(scheme=Standard9(n=8)).solve(T1)


def test_the_preset_has_flat_hparams_and_its_scheme_defaults():
    assert FDMixed().hparams == {
        "class_path": "maesolvers.FDMixed",
        "init_args": {
            "n": 64,
            "angles": None,
            "spline_degree": 3,
            "max_iter": 100,
            "tol": 1e-6,
        },
    }
    assert FDMixed().scheme == MixedHJB()
    assert isinstance(FDMixed().scheme.n, int)


@pytest.mark.parametrize(
    "solver",
    [
        FDMixed(n=10, angles=4),
        PolicyIteration(scheme=MixedHJB(n=10, seven_point=False), max_iter=9),
        Newton(scheme=MixedHJB(n=10)),
    ],
    ids=["preset", "composed", "newton"],
)
def test_hparams_rebuild_the_solver(solver):
    pytest.importorskip("jsonargparse")
    assert from_config(solver.hparams) == solver


def test_the_policy_matrix_is_sparse():
    grid = MixedHJB(n=10).discretize(T1)
    matrix, _ = MixedHJB(n=10).policy_system(grid, np.zeros(81))
    assert sp.issparse(matrix) and matrix.shape == (81, 81)
