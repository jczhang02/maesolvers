"""Standard9 on curved domains: Shortley-Weller stencils at the boundary (issue #1)."""

import dataclasses
import tracemalloc

import numpy as np
import pytest
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay

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
from maesolvers.metrics import evaluate, shared_grid

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


def test_the_operators_unpack_and_carry_names():
    ops = shortley_weller_operators(Disk((0.0, 0.0), 1.0), 8)
    nodes, interior, dxx, dyy, dxy, min_step = ops
    assert ops.nodes is nodes and ops.interior is interior and ops.dxx is dxx
    assert ops.dyy is dyy and ops.dxy is dxy and ops.min_step is min_step
    assert ops._fields == ("nodes", "interior", "dxx", "dyy", "dxy", "min_step")


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


def test_a_node_outside_the_circle_by_rounding_joins_the_boundary_once():
    """(0.6, 0.8) lies on the unit circle; rounding puts it outside."""
    nodes, interior, *_ = shortley_weller_operators(Disk((0, 0), 1), 10)
    on_grid = np.linspace(-1.0, 1.0, 11)[[8, 9]]
    assert not Disk((0, 0), 1).contains(on_grid)
    np.testing.assert_allclose(on_grid, [0.6, 0.8])
    near = np.flatnonzero(np.linalg.norm(nodes - on_grid, axis=1) < 1e-12)
    assert near.size == 1 and not interior[near[0]]


def _smooth(x):
    """A smooth function that is not a polynomial, to interpolate."""
    x = np.asarray(x, dtype=float)
    return np.exp(x[..., 0]) * np.cos(2.0 * x[..., 1])


@pytest.mark.parametrize(
    ("domain", "n"),
    [
        (Disk((0, 0), 1), 10),
        (Disk((0, 0), 1), 34),
        (Disk((0, 0), 1), 170),
        (Ellipse((0.2, -0.1), (0.7, 0.4)), 34),
        (Ellipse((0, 0), (1, 0.05)), 26),
        (Ellipse((0.5, 0.5), (0.5, 0.3)), 16),  # (0.5, 0.8) outside by rounding
        (CutQuarterDisk(), 70),  # boundary points 1e-5 steps apart
        (CutQuarterDisk(), 100),
    ],
    ids=lambda c: str(c),
)
def test_boundary_points_are_distinct_and_the_interpolant_is_finite(domain, n):
    """Points 1e-5 steps apart stay distinct; Qhull may drop one of them.

    At the two CutQuarterDisk cases an unknown lies some 3e-6 steps inside
    the arc, above the snap, and two of its crossings fall within 1e-5 steps
    of each other. Qhull leaves one out of the triangulation and its
    Clough-Tocher gradient is NaN, but then it is not a vertex of the hull,
    the only place where ``_scattered`` reads the gradients, and the
    interpolant stays finite.
    """
    nodes = shortley_weller_operators(domain, n).nodes
    assert len(np.unique(nodes, axis=0)) == len(nodes)
    values = _smooth(nodes)
    spline = CloughTocher2DInterpolator(nodes, values)
    grad = np.asarray(spline.grad).reshape(len(nodes), 2)
    assert np.isfinite(grad[np.unique(spline.tri.convex_hull)]).all()
    u = _scattered(nodes, values)
    np.testing.assert_allclose(u(nodes), values, rtol=0, atol=1e-12)
    assert np.isfinite(u(shared_grid(domain))).all()


EDGE_CASES = [
    (Ellipse((0.507, 0.076), (0.8, 0.4)), 18),
    (Ellipse((0.03, -0.77), (0.8, 0.65)), 100),
    (Disk((0.02, 0.51), 0.23), 106),
]
"""Domains whose bounding-box edges hold nodes that test inside by rounding."""


@pytest.mark.parametrize(("domain", "n"), EDGE_CASES, ids=lambda c: str(c))
def test_nodes_on_the_bounding_box_join_the_boundary(domain, n):
    box = domain.bbox
    x = np.linspace(box.lo[0], box.hi[0], n + 1)
    y = np.linspace(box.lo[1], box.hi[1], n + 1)
    frame = np.concatenate(
        [np.stack([np.full(n + 1, c), y], axis=-1) for c in x[[0, -1]]]
        + [np.stack([x, np.full(n + 1, c)], axis=-1) for c in y[[0, -1]]]
    )
    assert np.any(domain.interior(frame))  # rounding puts one inside
    nodes, interior, dxx, *_ = shortley_weller_operators(domain, n)
    assert dxx.shape == (np.count_nonzero(interior), len(nodes))
    inner = nodes[interior]
    assert np.all((inner > box.lo) & (inner < box.hi))
    on_frame = frame[domain.interior(frame)]
    assert all(np.any(np.all(nodes[~interior] == q, axis=1)) for q in on_frame)


def test_a_disk_with_a_node_on_its_box_converges_at_second_order():
    disk = _on(Disk((0.02, 0.51), 0.23), T1.exact, T1.f)
    errors = [evaluate(FDPoisson(n=n).solve(disk))["max"] for n in (26, 53, 106)]
    rates = np.log2(np.array(errors[:-1]) / errors[1:]) / np.log2([53 / 26, 2.0])
    assert np.all(rates > 1.8), rates


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


def test_the_interpolant_reaches_a_smooth_boundary_at_second_order():
    """Clough-Tocher inside the hull of the nodes, the slivers beyond it."""
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    rim = (1.0 - 1e-12) * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    inside = np.random.default_rng(0).uniform(-0.7, 0.7, (5, 10, 2))
    errors = []
    for n in (8, 16, 32):
        nodes, *_ = shortley_weller_operators(Disk((0, 0), 1), n)
        u = _scattered(nodes, _quadratic(nodes))
        assert u(inside).shape == (5, 10)
        np.testing.assert_allclose(u(inside), _quadratic(inside), atol=(2.0 / n) ** 2)
        errors.append(np.max(np.abs(u(rim) - _quadratic(rim))))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.8), errors


CORNERS = np.array([[0.5, 0.0], [(1 + np.sqrt(7)) / 4, (np.sqrt(7) - 1) / 4]])
"""The corners of T6's domain where the straight cut meets the axis and the arc."""


@pytest.mark.parametrize("exact", [_quadratic, T6.exact], ids=["quadratic", "T6"])
def test_the_interpolant_reaches_corners_off_the_grid_at_second_order(exact):
    """At a corner that is not a node the domain reaches past the hull by O(h)."""
    domain = CutQuarterDisk()
    rng = np.random.default_rng(1)
    for n in (16, 23, 32, 45, 64):
        h = 1.0 / n
        probes = np.concatenate(
            [c + 3 * h * rng.uniform(-1, 1, (2000, 2)) for c in CORNERS]
        )
        probes = np.concatenate(
            [probes, CORNERS + 1e-9 * np.array([[-0.3, 1.0], [-1.0, 0.2]])]
        )
        probes = probes[domain.interior(probes)]
        nodes, *_ = shortley_weller_operators(domain, n)
        u = _scattered(nodes, exact(nodes))
        past_hull = Delaunay(nodes).find_simplex(probes) < 0
        assert np.count_nonzero(past_hull) >= 2
        error = np.max(np.abs(u(probes) - exact(probes)))
        assert error < 0.5 * h**2, (n, error / h**2)


def test_the_interpolant_on_the_whole_bounding_box_stays_small_in_memory():
    nodes, *_ = shortley_weller_operators(Disk((0, 0), 1), 128)
    u = _scattered(nodes, _quadratic(nodes))
    axis = np.linspace(-1.0, 1.0, 200)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    tracemalloc.start()
    try:
        values = u(points)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert values.shape == (200, 200) and np.all(np.isfinite(values))
    assert peak < 100e6, peak


def test_the_interpolant_keeps_a_nan_value():
    nodes, *_ = shortley_weller_operators(Disk((0, 0), 1), 8)
    values = np.zeros(len(nodes))
    values[0] = np.nan
    u = _scattered(nodes, values)
    assert np.isnan(u(nodes[:1] + 1e-3))


@pytest.mark.parametrize("preset", [FDNewton, FDPoisson])
@pytest.mark.parametrize("n", [2, 8, 16, 33])
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
    assert result.info["interpolant"] == "clough_tocher"
    assert 0.0 < result.info["min_step"] < 1.0


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
