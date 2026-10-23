"""The second boundary value problem on a square grid and its preset FDTransport."""

from __future__ import annotations

import dataclasses
import warnings
from typing import ClassVar

import numpy as np
import pytest

from maesolvers import (
    SOLVERS,
    T1,
    T12,
    T16,
    Box,
    Disk,
    Ellipse,
    FDTransport,
    Newton,
    Problem,
    SecondBoundary,
    Transport,
    UnsupportedProblemError,
)
from maesolvers.classical.grid import BoxGrid
from maesolvers.classical.second_boundary import (
    admissible,
    directional_second_differences,
    face_normals,
    uniform_directions,
    upwind_gradient,
)
from maesolvers.metrics import gradient_fd, gradient_map_error

SQUARE = Box((-0.5, -0.5), (0.5, 0.5))


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


def _stretch(x):
    """``u = x1^2 + x2^2 / 2``: its gradient maps SQUARE onto ``[-1, 1] x [-0.5, 0.5]``."""
    x = np.asarray(x, dtype=float)
    return x[..., 0] ** 2 + 0.5 * x[..., 1] ** 2


STRETCH = Problem(
    "stretch",
    SQUARE,
    _constant(2.0),
    Transport(Box((-1.0, -0.5), (1.0, 0.5))),
    exact=_stretch,
)


@dataclasses.dataclass(frozen=True)
class _Rotated:
    """A target of one's own, ``M B_1``, known by its support function ``|M n|``."""

    matrix: tuple = ((0.6, 0.2), (0.2, 0.8))
    dim: ClassVar[int] = 2

    @property
    def bbox(self):
        r = np.linalg.norm(np.asarray(self.matrix), axis=1)
        return Box(tuple(-r), tuple(r))

    def _level(self, x):
        z = np.asarray(x) @ np.linalg.inv(np.asarray(self.matrix)).T
        return np.sum(z**2, axis=-1)

    def contains(self, x):
        return self._level(x) <= 1.0

    def interior(self, x):
        return self._level(x) < 1.0

    def support_function(self, n):
        return np.linalg.norm(np.asarray(n) @ np.asarray(self.matrix).T, axis=-1)


def _gradient_error(result, exact_gradient):
    """The paper's measure: centred differences at the interior nodes of the domain."""
    points = result.nodes[result.interior]
    h = result.info["h"][0]
    return gradient_map_error(
        gradient_fd(result.u, points, step=h), exact_gradient(points)
    )


# The kernels


def test_uniform_directions_are_unit_and_exact_on_the_axes():
    n = uniform_directions(8)
    assert n.shape == (8, 2)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    for axis in ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]):
        assert any(np.array_equal(row, axis) for row in n)
    with pytest.raises(ValueError, match="at least 4"):
        uniform_directions(3)


def test_admissible_directions_on_the_sides_and_at_the_corners():
    grid = BoxGrid(SQUARE, 4)
    ring = np.argwhere(~grid.interior.reshape(grid.shape))
    allowed = admissible(grid, face_normals())
    for (i, j), row in zip(ring, allowed, strict=True):
        expected = {
            k
            for k, hit in enumerate((i == 0, i == 4, j == 0, j == 4))
            if hit  # the faces the node lies on: lo x, hi x, lo y, hi y
        }
        assert set(np.flatnonzero(row)) == expected
    # Uniform directions: the open half-plane on a side, the closed quadrant
    # at a corner, so three of eight at both.
    counts = admissible(grid, uniform_directions(8)).sum(axis=1)
    assert np.all(counts == 3)


def test_upwind_gradient_is_exact_on_affine_functions_and_monotone():
    grid = BoxGrid(SQUARE, 6)
    directions = uniform_directions(16)
    owner, which = np.nonzero(admissible(grid, directions))
    nodes = np.flatnonzero(~grid.interior)[owner]
    operator = upwind_gradient(grid, nodes, directions[which])
    slope = np.array([0.3, -1.7])
    u = grid.nodes @ slope + 0.25
    assert np.allclose(operator @ u, directions[which] @ slope, atol=1e-12)
    dense = operator.toarray()
    rows = np.arange(len(nodes))
    assert np.all(dense[rows, nodes] >= 0.0)
    dense[rows, nodes] = 0.0
    assert np.all(dense <= 0.0)


def test_upwind_gradient_refuses_to_leave_the_grid_and_unequal_cells():
    grid = BoxGrid(SQUARE, 4)
    with pytest.raises(ValueError, match="leaves the grid"):
        upwind_gradient(grid, np.array([0]), np.array([[1.0, 0.0]]))
    flat = BoxGrid(Box((0.0, 0.0), (2.0, 1.0)), 4)
    with pytest.raises(ValueError, match="square cells"):
        upwind_gradient(flat, np.array([0]), np.array([[-1.0, 0.0]]))


def test_directional_second_differences_are_exact_on_quadratics():
    grid = BoxGrid(SQUARE, 6)
    u = _stretch(grid.nodes) + 0.4 * grid.nodes[:, 0] * grid.nodes[:, 1]
    hessian = np.array([[2.0, 0.4], [0.4, 1.0]])
    for direction in ((1, 0), (1, 1), (-1, 1), (2, 1), (-1, 3)):
        operator, full = directional_second_differences(grid, direction)
        nu = np.array(direction, dtype=float)
        expected = nu @ hessian @ nu / (nu @ nu)
        assert np.allclose((operator @ u)[full], expected)
        assert np.all((operator @ u)[~full] == 0.0)
        width = max(abs(k) for k in direction)
        assert np.all(full) == (width == 1)


# The scheme


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"n": 1}, ValueError, "n must be at least 2"),
        ({"interior": "standard"}, ValueError, "interior"),
        ({"stencil": 13}, ValueError, "9, 17 or 33"),
        ({"epsilon": 0.0}, ValueError, "epsilon"),
        ({"directions": 3}, ValueError, "directions must be at least 4"),
        ({"box": Box((0.0, 0.0), (2.0, 1.0))}, ValueError, "square"),
        ({"box": Disk((0.0, 0.0), 1.0)}, TypeError, "2D Box"),
        ({"jacobian_floor": -1.0}, ValueError, "jacobian_floor"),
        ({"spline_degree": 6}, ValueError, "spline_degree"),
    ],
)
def test_the_scheme_checks_its_parameters(change, error, match):
    with pytest.raises(error, match=match):
        SecondBoundary(**change)


def test_the_scheme_supports_transport_problems_in_a_square():
    scheme = SecondBoundary(n=8)
    assert scheme.supports(T16) and scheme.supports(T12)
    assert not scheme.supports(T1)  # Dirichlet
    ellipse = Problem(
        "ellipse",
        Ellipse((0.0, 0.0), (0.8, 0.4)),
        _constant(1.375),
        Transport(_Rotated()),
    )
    assert not scheme.supports(ellipse)  # its bounding box is not a square
    assert SecondBoundary(n=8, box=Box((-1.0, -1.0), (1.0, 1.0))).supports(ellipse)
    small = SecondBoundary(n=8, box=Box((-0.5, -0.5), (0.5, 0.5)))
    assert not small.supports(ellipse)  # the square does not hold the domain
    unknown = Problem("t", SQUARE, _constant(1.0), Transport(Ellipse((0, 0), (1, 1))))
    assert scheme.supports(unknown)
    with pytest.raises(UnsupportedProblemError, match="SecondBoundary"):
        scheme.discretize(T1)


def test_the_scheme_needs_a_nonnegative_f():
    negative = dataclasses.replace(STRETCH, f=_constant(-1.0))
    with pytest.raises(ValueError, match="f >= 0"):
        SecondBoundary(n=4).discretize(negative)


@pytest.mark.parametrize("interior", ["filtered", "monotone"])
@pytest.mark.parametrize("stencil", [9, 17])
def test_the_jacobian_is_the_derivative_of_the_residual(interior, stencil):
    scheme = SecondBoundary(n=8, interior=interior, stencil=stencil)
    grid = scheme.discretize(T16)
    rng = np.random.default_rng(0)
    u = T16.exact(grid.box_grid.nodes) + 1e-3 * rng.standard_normal(81)
    step = rng.standard_normal(81)
    eps = 1e-7
    difference = (
        scheme.residual(grid, u + eps * step) - scheme.residual(grid, u - eps * step)
    ) / (2.0 * eps)
    assert np.allclose(scheme.jacobian(grid, u) @ step, difference, atol=1e-4)


def test_the_start_solves_its_neumann_problem():
    scheme = SecondBoundary(n=8)
    grid = scheme.discretize(STRETCH)
    rhs = 2.0 * np.sqrt(scheme.rhs(grid))
    u = scheme.solve_laplace(grid, rhs)
    laplace = (grid.dxx + grid.dyy) @ u
    k = laplace / rhs[grid.interior]
    assert np.allclose(k, k[0])  # one constant for the compatibility
    assert np.allclose(grid.start_operator @ u, grid.start)
    assert u[grid.anchor] == pytest.approx(0.0, abs=1e-14)
    # The start map of a box onto a box is the exact map; the one-sided
    # Neumann rows put the start within O(h) of the exact solution.
    assert np.ptp(u - _stretch(grid.box_grid.nodes)) < grid.box_grid.h[0]


def test_newton_solves_a_quadratic_to_first_order_in_the_gradient():
    errors = []
    for n in (8, 16, 32):
        result = Newton(scheme=SecondBoundary(n=n)).solve(STRETCH)
        assert result.info["converged"]
        assert result.info["residual_max"] < 1e-10
        errors.append(
            _gradient_error(result, lambda x: x * np.array([2.0, 1.0]))["gradient_max"]
        )
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.1)


def test_the_solution_is_reported_on_the_nodes_of_the_domain():
    scheme = SecondBoundary(n=8)
    result = Newton(scheme=scheme).solve(dataclasses.replace(T12))
    box_grid = BoxGrid(T12.domain.bbox, 8)
    inside = T12.domain.contains(box_grid.nodes)
    assert np.array_equal(result.nodes, box_grid.nodes[inside])
    assert np.array_equal(result.interior, box_grid.interior[inside])
    assert np.all(np.isnan(result.node_hessian[~result.interior]))
    assert result.info["unknowns"] == 81
    assert {
        "h",
        "residual_max",
        "u_anchor",
        "filter_width",
        "monotone_fraction",
    } <= set(result.info)
    assert result.info["filter_width"] == pytest.approx(np.sqrt(0.25) + np.pi / 4)
    monotone = Newton(scheme=SecondBoundary(n=8, interior="monotone")).solve(T16)
    assert "filter_width" not in monotone.info


def test_a_target_of_ones_own_by_its_support_function():
    # BFO 2014, Section 6.2: the ellipse M_X B_1 onto M_Y B_1.
    mx, my = np.diag([0.8, 0.4]), np.array([[0.6, 0.2], [0.2, 0.8]])
    inverse = np.linalg.inv(mx) @ np.linalg.inv(my)
    turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    theta = np.arctan2(np.trace(inverse @ turn), np.trace(inverse))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    linear = my @ rotation @ np.linalg.inv(mx)
    assert np.allclose(linear, linear.T)
    problem = Problem(
        "ellipse",
        Ellipse((0.0, 0.0), (0.8, 0.4)),
        _constant(np.linalg.det(my) / np.linalg.det(mx)),
        Transport(_Rotated()),
    )
    preset = FDTransport(n=32, directions=64, box=Box((-1.0, -1.0), (1.0, 1.0)))
    with pytest.raises(TypeError, match="volume"):
        preset.solve(problem)  # the mass balance needs the target's volume
    composed = Newton(scheme=preset.scheme, start=preset.start, max_iter=200)
    result = composed.solve(problem)
    assert result.info["converged"]
    error = _gradient_error(result, lambda x: x @ linear.T)
    assert error["gradient_max"] < 0.1  # 0.0549 (BFO 2014, Table 2: 0.0667)


# The preset


def test_the_preset_is_newton_on_the_filtered_scheme_from_the_monotone_root():
    preset = FDTransport(n=16, directions=8, box=Box((-1.0, -1.0), (1.0, 1.0)))
    assert preset.scheme == SecondBoundary(
        n=16, directions=8, box=Box((-1.0, -1.0), (1.0, 1.0))
    )
    assert preset.start == Newton(
        scheme=SecondBoundary(
            n=16, interior="monotone", directions=8, box=Box((-1.0, -1.0), (1.0, 1.0))
        ),
        max_iter=200,
    )
    assert SOLVERS["FDTransport"] is FDTransport
    assert set(preset.hparams["init_args"]) == {
        "n",
        "stencil",
        "epsilon",
        "directions",
        "box",
        "jacobian_floor",
        "spline_degree",
        "max_iter",
        "tol",
    }
    result = FDTransport(n=16).solve(T16)
    composed = Newton(scheme=SecondBoundary(n=16), start=FDTransport(n=16).start)
    assert np.array_equal(result.values, composed.solve(T16).values)
    assert result.info["start_converged"] and result.info["converged"]


def test_the_preset_checks_the_mass_balance():
    unbalanced = dataclasses.replace(T16, f=lambda x: 2.0 * T16.f(x))
    with pytest.raises(ValueError, match="out of mass balance"):
        FDTransport(n=8).solve(unbalanced)
    with pytest.raises(UnsupportedProblemError):
        FDTransport(n=8).solve(T1)


def test_the_preset_reaches_t12_and_t16():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for problem in (T16, T12):
            result = FDTransport(n=32).solve(problem)
            assert result.info["converged"], problem.name
    t16 = _gradient_error(
        FDTransport(n=32).solve(T16), lambda x: gradient_fd(T16.exact, x)
    )
    # Within 3 % of BFO 2014, Table 1 (0.0220, 0.0127) at N = 32.
    assert t16["gradient_max"] == pytest.approx(0.02137, abs=5e-5)
    assert t16["gradient_rms"] == pytest.approx(0.01264, abs=5e-5)
