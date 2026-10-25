"""The transport helpers: target geometry, mass balance and the gradient-map error."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from scipy.spatial import cKDTree

from maesolvers import (
    T1,
    T12,
    T16,
    Box,
    CutQuarterDisk,
    Disk,
    Ellipse,
    FDNewton,
    Problem,
    Solution,
    Solver,
    Trace,
    Transport,
    UnsupportedProblemError,
)
from maesolvers.core import (
    can_check_mass_balance,
    check_mass_balance,
    closest_boundary_point,
    integrate,
    knows_support_function,
    signed_distance,
    support_function,
    volume,
)
from maesolvers.metrics import gradient_fd, gradient_map_error, shared_grid

TARGETS = [
    Box((-0.5, -0.5), (0.5, 0.5)),
    Box((0.0, 1.0), (2.0, 1.5)),
    Disk((1.0, 2.0), 0.7),
    Ellipse((3.5, 0.0), (2.0, 0.5)),  # T12's target, aspect ratio 4
    Ellipse((0.0, -1.0), (0.3, 1.2)),  # the long axis along y
    Ellipse((0.0, 0.0), (1.0, 1.0)),  # a circle
]


def _around(target, count, seed):
    """Points inside, outside and near the target, in a box three times its size."""
    box = target.bbox
    lo, hi = np.asarray(box.lo), np.asarray(box.hi)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return np.random.default_rng(seed).uniform(
        mid - 3 * half, mid + 3 * half, (count, 2)
    )


def _unit_circle(count):
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def _parametric_boundary(target, count):
    """Points on the target's boundary, dense in its parameter."""
    circle = _unit_circle(count)
    if isinstance(target, Box):
        lo, hi = np.asarray(target.lo), np.asarray(target.hi)
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        square = circle / np.max(np.abs(circle), axis=-1, keepdims=True)
        return mid + half * square
    if isinstance(target, Disk):
        return np.asarray(target.center) + target.radius * circle
    return np.asarray(target.center) + np.asarray(target.semi_axes) * circle


def _exact_gradient_t16(x):
    """The gradient map printed in Benamou-Froese-Oberman 2014, Section 6.1."""

    def q(z):
        a = -(z**2) / (8 * np.pi) + 1 / (256 * np.pi**3) + 1 / (32 * np.pi)
        return a * np.cos(8 * np.pi * z) + z * np.sin(8 * np.pi * z) / (32 * np.pi**2)

    def dq(z):
        return (z**2 - 0.25) * np.sin(8 * np.pi * z)

    x1, x2 = x[..., 0], x[..., 1]
    return np.stack([x1 + 4 * dq(x1) * q(x2), x2 + 4 * q(x1) * dq(x2)], axis=-1)


def _exact_gradient_t12(x):
    """``(2 x1 + 7/2, x2 / 2)``, Caboussat-Peruso 2025, Section 4.2."""
    return np.stack([2 * x[..., 0] + 3.5, 0.5 * x[..., 1]], axis=-1)


# support function


def test_support_function_of_each_target_in_closed_form():
    n = np.array([[1.0, 0.0], [0.0, -1.0], [3.0, 4.0]])
    box = Box((0.0, 1.0), (2.0, 1.5))
    assert np.allclose(support_function(box, n), [2.0, -1.0, 6.0 + 6.0])
    disk = Disk((1.0, 2.0), 0.5)
    assert np.allclose(support_function(disk, n), [1.5, -1.5, 11.0 + 2.5])
    ellipse = Ellipse((3.5, 0.0), (2.0, 0.5))
    assert np.allclose(support_function(ellipse, n), [5.5, 0.5, 10.5 + np.hypot(6, 2)])


@pytest.mark.parametrize("target", TARGETS, ids=repr)
def test_support_function_is_the_largest_projection_of_the_boundary(target):
    boundary = _parametric_boundary(target, 20000)
    n = _unit_circle(64) * np.linspace(0.5, 2.0, 64)[:, None]
    brute = np.max(n @ boundary.T, axis=-1)
    assert np.allclose(support_function(target, n), brute, atol=1e-6)


def test_support_function_of_a_box_in_three_dimensions():
    box = Box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert support_function(box, np.array([1.0, -1.0, 1.0])) == 4.0


# signed distance and the nearest boundary point


@pytest.mark.parametrize("target", TARGETS, ids=repr)
def test_signed_distance_is_the_distance_to_the_boundary_with_a_sign(target):
    y = _around(target, 3000, seed=0)
    d = signed_distance(target, y)
    brute, _ = cKDTree(_parametric_boundary(target, 400_000)).query(y)
    assert np.allclose(np.abs(d), brute, atol=1e-6)
    assert np.array_equal(d < 0, target.interior(y))


@pytest.mark.parametrize("target", TARGETS, ids=repr)
def test_signed_distance_is_lemma_2_1_of_bfo_2014(target):
    """``H(y) = max_{|n| = 1} (y . n - H*(n))`` for a convex target."""
    y = _around(target, 2000, seed=1)
    n = _unit_circle(8192)
    lemma = np.max(y @ n.T - support_function(target, n), axis=-1)
    # The directions are 2 pi / 8192 apart: the maximum misses by O(dtheta^2).
    assert np.allclose(signed_distance(target, y), lemma, atol=1e-6)


@pytest.mark.parametrize("target", TARGETS, ids=repr)
def test_the_normal_is_the_unit_gradient_of_the_signed_distance(target):
    y = _around(target, 2000, seed=2)
    foot, normal = closest_boundary_point(target, y)
    assert np.allclose(np.linalg.norm(normal, axis=-1), 1.0)
    assert np.allclose(signed_distance(target, foot), 0.0, atol=1e-12)
    assert np.allclose(np.sum((y - foot) * normal, axis=-1), signed_distance(target, y))
    grad = gradient_fd(lambda p: signed_distance(target, p), y)
    assert np.allclose(grad, normal, atol=1e-6)


@pytest.mark.parametrize("target", TARGETS[3:], ids=repr)
def test_the_ellipse_on_its_axes_and_at_its_centre(target):
    """Where the nearest point jumps, or the foot point equation has a pole."""
    c, (a, b) = np.asarray(target.center), target.semi_axes
    long, short = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))[:: 1 if a >= b else -1]
    big, small = max(a, b), min(a, b)
    y = c + np.array(
        [0 * long, 0.5 * big * long, -2 * big * long, 3 * small * short, big * long]
    )
    expected = [-small, None, big, 2 * small, 0.0]
    d = signed_distance(target, y)
    brute, _ = cKDTree(_parametric_boundary(target, 400_000)).query(y)
    assert np.allclose(np.abs(d), brute, atol=1e-6)
    for value, want in zip(d, expected, strict=True):
        if want is not None:
            assert value == pytest.approx(want, abs=1e-12)


def _ellipse_distance_by_angle(target, y):
    """Distance from each point to the ellipse, by search over its parameter.

    The nearest of 4096 boundary points, refined by ternary search on the
    angle around it, accurate to about 1e-14 where the nearest point is
    unique and to the gap between the two nearest points where it is not.
    """
    c, axes = np.asarray(target.center), np.asarray(target.semi_axes)

    def dist(theta):
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        return np.linalg.norm(c + axes * circle - y, axis=-1)

    coarse = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    step = coarse[1]
    best = coarse[np.argmin(dist(coarse[:, None]), axis=0)]
    low, high = best - step, best + step
    for _ in range(100):
        left, right = (2 * low + high) / 3, (low + 2 * high) / 3
        closer = dist(left) < dist(right)
        low, high = np.where(closer, low, left), np.where(closer, right, high)
    return dist((low + high) / 2)


@pytest.mark.parametrize("target", TARGETS[3:], ids=repr)
@pytest.mark.parametrize("off_axis", [1e-300, 1e-20, 1e-17, 1e-15, 1e-13, 1e-9])
def test_the_ellipse_near_its_major_axis(target, off_axis):
    """Accurate and finite at any distance from the major axis.

    Along the major axis, inside the evolute (the foot leaves the axis),
    past it, near the vertex and outside; above and below the axis.
    """
    c, (a, b) = np.asarray(target.center), target.semi_axes
    long, short = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))[:: 1 if a >= b else -1]
    big, small = max(a, b), min(a, b)
    evolute = (big**2 - small**2) / big
    along = np.array([0.1, 0.5, 0.95 * evolute, 1.05 * evolute, 0.999, 1.5, 3.0])
    along = np.concatenate([along[:2], along[2:4] if evolute > 0 else [], along[4:]])
    shift = along[:, None] * big * long
    y = c + np.concatenate([shift + off_axis * short, -shift - off_axis * short])
    foot, normal = closest_boundary_point(target, y)
    d = signed_distance(target, y)
    assert np.all(np.isfinite(foot)) and np.all(np.isfinite(normal))
    assert np.all(np.isfinite(d))
    assert np.allclose(np.sum(((foot - c) / [a, b]) ** 2, axis=-1), 1.0, atol=1e-14)
    assert np.allclose(np.abs(d), _ellipse_distance_by_angle(target, y), atol=1e-10)
    assert np.array_equal(d < 0, target.interior(y))


def test_the_ellipse_near_its_centre():
    """Near the centre of a circle written as an ellipse, and of a near-circle."""
    y = np.array([[1e-17, 1e-17], [1e-10, 1e-17], [-1e-300, 1e-17]])
    for target in (
        Ellipse((0.0, 0.0), (1.0, 1.0)),
        Ellipse((0.0, 0.0), (1 + 1e-9, 1.0)),
    ):
        foot, normal = closest_boundary_point(target, y)
        assert np.all(np.isfinite(foot)) and np.all(np.isfinite(normal))
        d = signed_distance(target, y)
        assert np.allclose(np.abs(d), _ellipse_distance_by_angle(target, y), atol=1e-10)


def test_signed_distance_against_boundary_distance_inside():
    """Inside, the signed distance is minus the domain's boundary distance."""
    for domain in (Box((0.0, 0.0), (1.0, 2.0)), Disk((0.5, 0.0), 2.0)):
        y = shared_grid(domain, size=41)
        assert np.allclose(-signed_distance(domain, y), domain.boundary_distance(y))


def test_signed_distance_of_a_box_in_three_dimensions():
    box = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    y = np.array([[0.5, 0.5, 0.2], [2.0, 2.0, 2.0], [0.5, 1.0, 0.5]])
    assert np.allclose(signed_distance(box, y), [-0.2, np.sqrt(3.0), 0.0])


def test_the_helpers_refuse_another_target_and_a_wrong_shape():
    for function in (signed_distance, support_function, closest_boundary_point):
        with pytest.raises(TypeError, match="knows Box, Disk and Ellipse"):
            function(CutQuarterDisk(), np.zeros((3, 2)))
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 2\)"):
            function(Disk((0.0, 0.0), 1.0), np.zeros((3, 3)))
    with pytest.raises(TypeError, match="knows Box, Disk and Ellipse"):
        volume(CutQuarterDisk())


@dataclass(frozen=True)
class _Rotated:
    """A target of one's own: the ellipse ``M B_1``, ``M`` symmetric, by its support."""

    matrix: tuple = ((0.6, 0.2), (0.2, 0.8))
    width: int = 0  # extra trailing axis in the output, to test the shape check

    dim: ClassVar[int] = 2
    bbox: ClassVar[Box] = Box((-1.0, -1.0), (1.0, 1.0))

    def contains(self, x):
        return np.ones(np.shape(x)[:-1], dtype=bool)

    def interior(self, x):
        return self.contains(x)

    def support_function(self, n):
        out = np.linalg.norm(np.asarray(n) @ np.asarray(self.matrix).T, axis=-1)
        return out[..., None].repeat(self.width, -1) if self.width else out


def test_support_function_asks_a_target_of_ones_own():
    from maesolvers.core import SupportFunction

    target = _Rotated()
    assert isinstance(target, SupportFunction)
    theta = np.linspace(0.0, 2.0 * np.pi, 7)
    n = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    # The boundary points M (cos t, sin t); the largest projection on n.
    t = np.linspace(0.0, 2.0 * np.pi, 20001)
    boundary = np.stack([np.cos(t), np.sin(t)], axis=-1) @ np.asarray(target.matrix).T
    assert np.allclose(
        support_function(target, n), np.max(n @ boundary.T, axis=1), atol=1e-7
    )
    with pytest.raises(ValueError, match="returned shape"):
        support_function(_Rotated(width=2), n)
    # The other helpers keep to the shapes they know.
    with pytest.raises(TypeError, match="knows Box, Disk and Ellipse"):
        signed_distance(target, n)


def test_the_predicates_name_what_the_helpers_know():
    target = _Rotated()
    for known in (*TARGETS, target):
        assert knows_support_function(known)
    assert not knows_support_function(CutQuarterDisk())
    assert can_check_mass_balance(T12) and can_check_mass_balance(T16)
    assert not can_check_mass_balance(T1)  # Dirichlet
    square = Box((-1.0, -1.0), (1.0, 1.0))
    own_target = Problem("t", square, lambda x: x[..., 0], Transport(target))
    own_domain = Problem("d", CutQuarterDisk(), lambda x: x[..., 0], Transport(square))
    assert not can_check_mass_balance(own_target)
    assert not can_check_mass_balance(own_domain)


# volume, integrate and the mass balance


def test_volumes():
    assert volume(Box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))) == 6.0
    assert volume(Disk((1.0, 1.0), 2.0)) == pytest.approx(4 * np.pi)
    assert volume(Ellipse((3.5, 0.0), (2.0, 0.5))) == pytest.approx(np.pi)


@pytest.mark.parametrize(
    "domain",
    [Box((0.0, -1.0), (2.0, 0.5)), Disk((1.0, 2.0), 0.7), TARGETS[4]],
    ids=repr,
)
def test_integrate_is_exact_for_a_quadratic(domain):
    c = np.asarray(domain.bbox.lo) / 2 + np.asarray(domain.bbox.hi) / 2

    def f(x):
        return 1.0 + np.sum((x - c) ** 2, axis=-1)

    if isinstance(domain, Box):
        sides = np.subtract(domain.hi, domain.lo)
        exact = np.prod(sides) * (1.0 + np.sum(sides**2) / 12)
    else:
        a, b = (domain.radius,) * 2 if isinstance(domain, Disk) else domain.semi_axes
        exact = np.pi * a * b * (1.0 + (a**2 + b**2) / 4)
    assert integrate(domain, f, points=8) == pytest.approx(exact, rel=1e-13)


@pytest.mark.parametrize("problem", [T12, T16], ids=lambda p: p.name)
def test_the_catalogue_transport_problems_are_in_mass_balance(problem):
    check_mass_balance(problem, rtol=1e-12)
    target = problem.boundary.target
    assert integrate(problem.domain, problem.f) == pytest.approx(volume(target), 1e-13)


def _unbalanced(density):
    return Problem(
        "unbalanced",
        Disk((0.0, 0.0), 1.0),
        lambda x: np.full(np.shape(x)[:-1], density),
        Transport(Ellipse((3.5, 0.0), (2.0, 0.5))),
    )


def test_an_unbalanced_problem_fails_the_check():
    with pytest.raises(ValueError, match=r"out of mass balance.*6\.28318530718"):
        check_mass_balance(_unbalanced(2.0))
    check_mass_balance(_unbalanced(1.0005))  # within the default rtol of 1e-3
    with pytest.raises(ValueError, match=r"rtol = 0\.0001"):
        check_mass_balance(_unbalanced(1.0005), rtol=1e-4)


def _step_density(x):
    """Balanced on the unit square, with a jump at x_0 = 0.3."""
    return np.where(x[..., 0] < 0.3, 0.5 / 0.3, 0.5 / 0.7)


def test_a_balanced_density_that_jumps_needs_a_wider_rtol_or_its_mass():
    """The quadrature resolves the step to about 1e-2, as documented."""
    square = Box((0.0, 0.0), (1.0, 1.0))
    step = Problem("step", square, _step_density, Transport(square))
    assert 2e-3 < abs(integrate(square, _step_density) - 1.0) < 1e-2
    with pytest.raises(ValueError, match="out of mass balance"):
        check_mass_balance(step)
    check_mass_balance(step, rtol=2e-2)
    check_mass_balance(step, mass=1.0)


def test_the_mass_balance_on_another_domain_takes_the_mass():
    cut = Problem("cut", CutQuarterDisk(), T12.f, Transport(Disk((0.0, 0.0), 1.0)))
    with pytest.raises(TypeError, match="knows Box, Disk and Ellipse"):
        check_mass_balance(cut)
    check_mass_balance(cut, mass=np.pi)
    with pytest.raises(ValueError, match="out of mass balance"):
        check_mass_balance(cut, mass=3.0)


def test_integrate_in_slabs_on_a_box_in_four_dimensions():
    """``points ** d`` evaluations, at most about a million at a time."""
    box = Box((0.0,) * 4, (1.0, 2.0, 1.0, 0.5))
    sizes = []

    def f(x):
        sizes.append(x[..., 0].size)
        return np.sum(x, axis=-1)

    # The integral of x_0 + ... + x_3 is the volume times the sum of the midpoints.
    exact = 1.0 * (0.5 + 1.0 + 0.5 + 0.25)
    assert integrate(box, f, points=40) == pytest.approx(exact, rel=1e-13)
    assert sum(sizes) == 40**4 and max(sizes) <= 2**20


def test_the_mass_balance_needs_a_transport_problem():
    from maesolvers import T1

    with pytest.raises(ValueError, match="not Transport"):
        check_mass_balance(T1)


@dataclass(frozen=True, kw_only=True)
class _Transporter(Solver):
    domains: ClassVar = (Disk, Box)
    boundaries: ClassVar = (Transport,)

    def check(self, problem):
        super().check(problem)
        check_mass_balance(problem)

    def setup(self, problem):
        return None

    def run(self, problem, context, callbacks):
        return Trace(None, 0, True)

    def finalize(self, problem, context, state):
        return Solution(u=lambda x: np.zeros(np.shape(x)[:-1]))


def test_a_transport_solver_checks_the_mass_balance_before_it_solves():
    assert _Transporter().solve(T12).info["converged"]
    assert _Transporter().solve(T16).info["converged"]
    with pytest.raises(ValueError, match="out of mass balance"):
        _Transporter().solve(_unbalanced(2.0))
    with pytest.raises(UnsupportedProblemError):
        _Transporter().solve(
            Problem("cut", CutQuarterDisk(), T12.f, Transport(Disk((0, 0), 1)))
        )


def test_the_base_check_leaves_the_mass_balance_to_the_solver():
    """``super().check`` neither integrates f nor needs to know the target."""

    @dataclass(frozen=True)
    class Unchecked(_Transporter):
        def check(self, problem):
            Solver.check(self, problem)

    assert Unchecked().solve(_unbalanced(2.0)).info["converged"]
    polygon = Problem(
        "polygon target",
        Box((0.0, 0.0), (1.0, 1.0)),
        T16.f,
        Transport(CutQuarterDisk()),
    )
    assert Unchecked().solve(polygon).info["converged"]


# the gradient-map error


@pytest.mark.parametrize(
    ("problem", "exact"),
    [(T12, _exact_gradient_t12), (T16, _exact_gradient_t16)],
    ids=["T12", "T16"],
)
def test_the_exact_solution_has_no_gradient_map_error(problem, exact):
    points = shared_grid(problem.domain, size=101)
    errors = gradient_map_error(gradient_fd(problem.exact, points), exact(points))
    assert set(errors) == {"gradient_max", "gradient_rms"}
    assert errors["gradient_max"] < 1e-8
    assert errors["gradient_rms"] <= errors["gradient_max"]


@pytest.mark.parametrize(
    ("problem", "exact"),
    [(T12, _exact_gradient_t12), (T16, _exact_gradient_t16)],
    ids=["T12", "T16"],
)
def test_the_exact_gradient_maps_the_domain_onto_the_target(problem, exact):
    target = problem.boundary.target
    inside = signed_distance(target, exact(shared_grid(problem.domain, size=101)))
    assert np.all(inside <= 1e-12)
    on_boundary = _parametric_boundary(problem.domain, 2000)
    assert np.allclose(signed_distance(target, exact(on_boundary)), 0.0, atol=1e-12)


def test_the_gradient_map_error_is_the_norm_of_the_difference():
    gradient = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    errors = gradient_map_error(gradient, np.zeros((3, 2)))
    assert errors == {
        "gradient_max": 5.0,
        "gradient_rms": pytest.approx(np.sqrt(26 / 3)),
    }
    nan = gradient_map_error(np.array([[np.nan, 0.0]]), np.zeros((1, 2)))
    assert np.isnan(nan["gradient_max"]) and np.isnan(nan["gradient_rms"])


def test_the_gradient_map_error_refuses_mismatched_or_no_points():
    with pytest.raises(ValueError, match="same shape"):
        gradient_map_error(np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="at least one point"):
        gradient_map_error(np.zeros((0, 2)), np.zeros((0, 2)))


def test_gradient_fd_with_a_given_step():
    def u(x):
        return np.sum(x**3, axis=-1)

    x = np.array([[0.5, -1.0, 2.0]])
    assert np.allclose(gradient_fd(u, x), 3 * x**2, atol=1e-8)
    # Central differences of a cubic err by step^2 exactly.
    assert np.allclose(gradient_fd(u, x, step=1e-2), 3 * x**2 + 1e-4)
    with pytest.raises(ValueError, match="step must be positive"):
        gradient_fd(u, x, step=0.0)


def test_gradient_fd_of_a_grid_result_on_the_interior_points():
    # Past the box a classical result's u (the 2D spline, constant there) is
    # not the solution, so the contract asks for interior points: there the
    # error is the scheme's, 2.0e-3 at n = 16.
    result = FDNewton(n=16).solve(T1)
    points = shared_grid(T1.domain)
    points = points[T1.domain.interior(points)]
    errors = gradient_map_error(
        gradient_fd(result.u, points), gradient_fd(T1.exact, points)
    )
    assert errors["gradient_max"] < 3e-3
