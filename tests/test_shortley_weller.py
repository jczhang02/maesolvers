"""Standard9 on curved domains: Shortley-Weller stencils at the boundary (issue #1)."""

import dataclasses

import numpy as np
import pytest

from maesolvers import (
    T1,
    T6,
    T8,
    Box,
    CutQuarterDisk,
    Dirichlet,
    Disk,
    Ellipse,
    FDNewton,
    FDPoisson,
    Newton,
    Problem,
    Standard9,
)
from maesolvers.classical.schemes import _scattered, shortley_weller_operators
from maesolvers.metrics import evaluate

DOMAINS = [
    Disk((0.5, 0.5), 0.5),
    Ellipse((0.2, -0.1), (0.7, 0.4)),
    CutQuarterDisk(),
]


def _quadratic(x):
    """``u = x^2 + x y / 2 + 3 y^2 / 2 - y``: u_xx = 2, u_yy = 3, u_xy = 1/2."""
    x = np.asarray(x, dtype=float)
    return (
        x[..., 0] ** 2 + 0.5 * x[..., 0] * x[..., 1] + 1.5 * x[..., 1] ** 2 - x[..., 1]
    )


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


def _on(domain, exact, f):
    return Problem("Q", domain, f, Dirichlet(exact), exact=exact)


T1_ON_DISK = _on(Disk((0.5, 0.5), 0.5), T1.exact, T1.f)
"""T1's exact solution on the disk inscribed in its square."""


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda d: type(d).__name__)
def test_the_stencils_are_exact_on_quadratics(domain):
    scheme = Standard9(n=12)
    grid = scheme.discretize(_on(domain, _quadratic, _constant(5.75)))
    v = _quadratic(grid.nodes[grid.interior])
    assert np.min(grid.row_scale) < 1.0  # some stencils are cut
    for d2, expected in zip(
        scheme.second_derivatives(grid, v), (2.0, 3.0, 0.5), strict=True
    ):
        np.testing.assert_allclose(d2, expected, rtol=1e-8)
    np.testing.assert_allclose(scheme.residual(grid, v), 0.0, atol=1e-8)


def test_full_stencils_are_the_nine_point_stencils():
    nodes, interior, dxx, dyy, dxy, min_step = shortley_weller_operators(
        Disk((0.0, 0.0), 1.0), 16
    )
    h = 2.0 / 16

    def u(p):
        return np.sin(p[..., 0] + 2.0 * p[..., 1])

    full = min_step == 1.0
    p = nodes[interior][full]
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    expected = (
        (u(p + ex) - 2 * u(p) + u(p - ex)) / h**2,
        (u(p + ey) - 2 * u(p) + u(p - ey)) / h**2,
        (u(p + ex + ey) - u(p + ex - ey) - u(p - ex + ey) + u(p - ex - ey))
        / (4 * h**2),
    )
    for d, e in zip((dxx, dyy, dxy), expected, strict=True):
        np.testing.assert_allclose((d @ u(nodes))[full], e, atol=1e-11)
    assert np.all(np.diff(dxy[np.flatnonzero(full)].indptr) == 4)


def test_boundary_points_lie_on_the_circle_and_no_step_is_tiny():
    nodes, interior, *_, min_step = shortley_weller_operators(Disk((0, 0), 1), 34)
    radius = np.linalg.norm(nodes[~interior], axis=1)
    np.testing.assert_allclose(radius, 1.0, atol=1e-15)
    assert np.all(np.linalg.norm(nodes[interior], axis=1) < 1.0)
    assert len(min_step) == np.count_nonzero(interior)
    assert np.min(min_step) >= 1e-6 and np.max(min_step) == 1.0


def test_a_node_on_the_circle_joins_the_boundary():
    """(15/17, 8/17) lies on the unit circle; rounding puts it inside."""
    point = np.array([15.0, 8.0]) / 17.0
    nodes, interior, *_ = shortley_weller_operators(Disk((0, 0), 1), 34)
    on_grid = np.linspace(-1.0, 1.0, 35)[[32, 25]]
    assert Disk((0, 0), 1).interior(on_grid)  # the node is inside, by rounding
    np.testing.assert_allclose(on_grid, point)
    at = np.flatnonzero(np.all(nodes == on_grid, axis=1))
    assert at.size == 1 and not interior[at[0]]


def test_a_domain_without_grid_nodes_inside_is_an_error():
    @dataclasses.dataclass(frozen=True)
    class Thin:
        dim = 2
        bbox = Box((0.0, 0.0), (1.0, 1.0))

        def contains(self, x):
            return np.abs(np.asarray(x)[..., 0] - 0.3) < 0.01

        def interior(self, x):
            return np.abs(np.asarray(x)[..., 0] - 0.3) < 0.01

    with pytest.raises(ValueError, match="no node of the 3 x 3 grid"):
        shortley_weller_operators(Thin(), 3)


def test_jacobian_matches_finite_differences_of_the_scaled_residual():
    scheme = Standard9(n=6)
    grid = scheme.discretize(T1_ON_DISK)
    k = int(np.count_nonzero(grid.interior))
    assert np.min(grid.row_scale) < 0.5
    v = scheme.solve_laplace(grid, np.full(k, 2.0))
    jac = scheme.jacobian(grid, v).toarray()
    eps = 1e-6
    fd = np.column_stack(
        [
            (scheme.residual(grid, v + eps * e) - scheme.residual(grid, v - eps * e))
            / (2 * eps)
            for e in np.eye(k)
        ]
    )
    np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-5)


def test_the_residual_is_scaled_by_the_smallest_step():
    scheme = Standard9(n=10)
    grid = scheme.discretize(T1_ON_DISK)
    v = scheme.solve_laplace(grid, np.full(len(grid.f), 2.0))
    *_, min_step = shortley_weller_operators(T1_ON_DISK.domain, 10)
    np.testing.assert_array_equal(grid.row_scale, min_step)
    unscaled = np.prod(scheme.second_derivatives(grid, v)[:2], axis=0)
    unscaled -= scheme.second_derivatives(grid, v)[2] ** 2 + grid.f
    np.testing.assert_allclose(scheme.residual(grid, v), min_step * unscaled)
    assert np.all(Standard9(n=6).discretize(T1).row_scale == 1.0)


def test_the_interpolant_reaches_the_boundary_at_second_order():
    """Clough-Tocher inside the hull of the nodes, the nearest hull edge beyond it."""

    def linear(p):
        return 1.0 + 2.0 * p[..., 0] - 0.5 * p[..., 1]

    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    rim = (1.0 - 1e-12) * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    inside = np.random.default_rng(0).uniform(-0.7, 0.7, (5, 10, 2))
    errors = []
    for n in (8, 16, 32):
        nodes, *_ = shortley_weller_operators(Disk((0, 0), 1), n)
        u = _scattered(nodes, linear(nodes))
        assert u(inside).shape == (5, 10)
        np.testing.assert_allclose(u(inside), linear(inside), atol=1e-7)
        errors.append(np.max(np.abs(u(rim) - linear(rim))))
    assert errors[0] < 0.02
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.8), errors


def test_the_interpolant_keeps_a_nan_value():
    nodes, *_ = shortley_weller_operators(Disk((0, 0), 1), 8)
    values = np.zeros(len(nodes))
    values[0] = np.nan
    u = _scattered(nodes, values)
    assert np.isnan(u(nodes[:1] + 1e-3))


@pytest.mark.parametrize("preset", [FDNewton, FDPoisson])
@pytest.mark.parametrize("n", [8, 16, 33])
def test_t8_reaches_the_convex_root(preset, n):
    result = preset(n=n).solve(T8)
    m = evaluate(result)
    assert result.info["converged"]
    assert m["max"] < 1e-10
    assert m["nonconvex_fraction"] == 0.0
    assert m["eigmin"] == pytest.approx(1.0, abs=1e-8)
    assert result.info["h"] == pytest.approx((2.0 / n, 2.0 / n))
    assert np.all(np.isnan(result.node_hessian[~result.interior]))
    assert result.info["unknowns"] == np.count_nonzero(result.interior)


def test_newton_from_a_concave_start_reaches_the_concave_root():
    """Both roots solve the discrete equation; the start picks one."""

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ConcaveStart(Newton):
        def initial_guess(self, problem, context):
            rhs = -2.0 * np.sqrt(self.scheme.rhs(context))
            return self.scheme.solve_laplace(context, rhs)

    result = ConcaveStart(scheme=Standard9(n=16)).solve(T8)
    assert result.info["converged"]
    assert evaluate(result)["nonconvex_fraction"] == 1.0
    np.testing.assert_allclose(result.values, -T8.exact(result.nodes), atol=1e-12)


@pytest.mark.parametrize("preset", [FDNewton, FDPoisson])
@pytest.mark.parametrize(
    "problem",
    [T1_ON_DISK, _on(Ellipse((0.5, 0.5), (0.5, 0.3)), T1.exact, T1.f), T6],
    ids=["disk", "ellipse", "T6"],
)
def test_second_order_on_curved_domains(preset, problem):
    errors = []
    for n in (16, 32, 64):
        result = preset(n=n).solve(problem)
        assert result.info["converged"]
        errors.append(evaluate(result)["max"])
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 1.8), rates


def test_newton_meets_its_tolerance_next_to_short_steps():
    """Without the row scaling the rounding of the 1/t weights stalls Newton."""
    ellipse = _on(Ellipse((0.5, 0.5), (0.5, 0.3)), T1.exact, T1.f)
    result = FDNewton(n=128).solve(ellipse)
    assert result.info["converged"]
    assert np.min(Standard9(n=128).discretize(ellipse).row_scale) < 0.05
