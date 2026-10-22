"""The standard 9-point scheme and its pure difference operators."""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from maesolvers import T1, T8, T12, Box, Dirichlet, Problem, UnsupportedProblemError
from maesolvers.classical import Standard9
from maesolvers.classical.schemes import (
    difference_operators,
    monge_ampere,
    monge_ampere_jacobian,
)


def _quadratic(x):
    """``u = x^2 + x y / 2 + 3 y^2 / 2 - y``: u_xx = 2, u_yy = 3, u_xy = 1/2."""
    x = np.asarray(x, dtype=float)
    return (
        x[..., 0] ** 2 + 0.5 * x[..., 0] * x[..., 1] + 1.5 * x[..., 1] ** 2 - x[..., 1]
    )


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


QUADRATIC = Problem(
    "Q",
    Box((-1.0, 0.0), (2.0, 0.5)),
    _constant(2.0 * 3.0 - 0.25),
    Dirichlet(_quadratic),
    exact=_quadratic,
)


def test_difference_operators_are_exact_on_quadratics_with_unequal_spacing():
    n, hx, hy = 6, 0.5, 0.1
    x, y = np.arange(n + 1) * hx, np.arange(n + 1) * hy
    nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    dxx, dyy, dxy = difference_operators(n, hx, hy)
    assert dxx.shape == dyy.shape == dxy.shape == ((n - 1) ** 2, (n + 1) ** 2)
    u = _quadratic(nodes)
    np.testing.assert_allclose(dxx @ u, 2.0, rtol=1e-12)
    np.testing.assert_allclose(dyy @ u, 3.0, rtol=1e-12)
    np.testing.assert_allclose(dxy @ u, 0.5, rtol=1e-12)


def test_difference_operators_use_nine_points():
    dxx, dyy, dxy = difference_operators(4, 1.0, 1.0)
    stencil = (abs(dxx) + abs(dyy) + abs(dxy)).getrow(4)
    assert stencil.nnz == 9


def test_monge_ampere_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    dxx, dyy, dxy = (sp.csr_matrix(rng.standard_normal((5, 5))) for _ in range(3))
    v = rng.standard_normal(5)

    def det(w):
        return monge_ampere(dxx @ w, dyy @ w, dxy @ w)

    jac = monge_ampere_jacobian(dxx @ v, dyy @ v, dxy @ v, dxx, dyy, dxy).toarray()
    eps = 1e-6
    fd = np.column_stack(
        [(det(v + eps * e) - det(v - eps * e)) / (2 * eps) for e in np.eye(5)]
    )
    np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-8)


def test_the_exact_quadratic_has_zero_residual():
    scheme = Standard9(n=6)
    grid = scheme.discretize(QUADRATIC)
    v = _quadratic(grid.nodes[grid.interior])
    np.testing.assert_allclose(scheme.residual(grid, v), 0.0, atol=1e-11)
    for d2, expected in zip(scheme.second_derivatives(grid, v), (2.0, 3.0, 0.5)):
        np.testing.assert_allclose(d2, expected, rtol=1e-10)


def test_jacobian_matches_finite_differences_of_the_residual():
    scheme = Standard9(n=5)
    grid = scheme.discretize(T1)
    v = scheme.solve_laplace(grid, np.full(16, 2.0))
    jac = scheme.jacobian(grid, v).toarray()
    eps = 1e-6
    fd = np.column_stack(
        [
            (scheme.residual(grid, v + eps * e) - scheme.residual(grid, v - eps * e))
            / (2 * eps)
            for e in np.eye(16)
        ]
    )
    np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-5)


def test_solve_laplace_keeps_the_dirichlet_data():
    """The Laplacian of a quadratic is constant, and the scheme is exact on it."""
    scheme = Standard9(n=6)
    grid = scheme.discretize(QUADRATIC)
    v = scheme.solve_laplace(grid, np.full(25, 5.0))
    np.testing.assert_allclose(v, _quadratic(grid.nodes[grid.interior]), atol=1e-12)


def test_rhs_is_f_at_the_unknowns():
    scheme = Standard9(n=4)
    grid = scheme.discretize(T1)
    np.testing.assert_array_equal(scheme.rhs(grid), T1.f(grid.nodes[grid.interior]))


def test_solution_fills_the_boundary_and_leaves_its_hessian_undefined():
    scheme = Standard9(n=6, spline_degree=2)
    grid = scheme.discretize(QUADRATIC)
    v = _quadratic(grid.nodes[grid.interior])
    solution = scheme.solution(grid, v)
    np.testing.assert_allclose(solution.values, _quadratic(grid.nodes), atol=1e-14)
    assert np.all(np.isnan(solution.node_hessian[~grid.interior]))
    np.testing.assert_allclose(
        solution.node_hessian[grid.interior][0], [[2, 0.5], [0.5, 3]]
    )
    assert solution.unknowns == 25
    assert solution.info["h"] == pytest.approx((0.5, 1 / 12))
    assert solution.info["residual_max"] < 1e-11
    point = np.array([[0.3, 0.2]])
    assert solution.u(point) == pytest.approx(_quadratic(point), abs=1e-12)


def test_supports_a_2d_dirichlet_box_only():
    scheme = Standard9()
    assert scheme.supports(T1)
    assert not scheme.supports(T8) and not scheme.supports(T12)
    cube = Problem(
        "C", Box((0,) * 3, (1,) * 3), _constant(1.0), Dirichlet(_constant(0))
    )
    assert not scheme.supports(cube)
    with pytest.raises(UnsupportedProblemError, match="T8"):
        scheme.discretize(T8)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (dict(n=1), ValueError, "n must be at least 2"),
        (dict(n=8.5), TypeError, "n must be an integer"),
        (dict(spline_degree=0), ValueError, "spline_degree must be at least 1"),
        (dict(n=4, spline_degree=5), ValueError, r"min\(5, n\) = 4"),
        (dict(spline_degree="3"), TypeError, "spline_degree must be an integer"),
    ],
)
def test_standard9_validates_its_parameters(kwargs, error, message):
    with pytest.raises(error, match=message):
        Standard9(**kwargs)


def test_standard9_is_a_frozen_keyword_only_part():
    assert Standard9(n=np.int64(8)) == Standard9(n=8)
    assert type(Standard9(n=np.int64(8)).n) is int
    with pytest.raises(TypeError):
        Standard9(8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Standard9().n = 4
