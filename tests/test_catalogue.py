"""The T-id catalogue against values computed independently of its code.

Four independent checks:

- the point values of the shared test-set specification (Section 4), computed
  there with numpy float64 and printed to 12 significant digits;
- for every problem with an exact solution, ``det D^2 u`` by central finite
  differences of ``u`` (a stencil written here, not the library's) against
  ``f``, at random interior points away from the sets where ``u`` is not
  smooth; and the Dirichlet data against ``u`` on the boundary;
- values worked out by hand at special points of the catalogue-only problems;
- the gradient of ``u`` maps the domain onto a set whose area is the integral
  of ``f``: onto the target for the transport problems, and for T1 and the
  Dirichlet problems with a kink or a singular point, so that a singular part
  of the Monge-Ampere measure that ``f`` omits fails.

The last section runs every included solver on every catalogue problem.
"""

import numpy as np
import pytest
from scipy.special import erfi

import maesolvers
from maesolvers import FDNewton, FDPoisson
from maesolvers.catalogue import CATALOGUE, SHARED_TEST_SET
from maesolvers.core import (
    Box,
    CutQuarterDisk,
    Dirichlet,
    Disk,
    Result,
    Transport,
    UnsupportedProblemError,
)
from maesolvers.metrics import evaluate, shared_grid

# The catalogue as a whole


def test_catalogue_is_keyed_by_t_id():
    assert list(CATALOGUE) == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T12",
        "T16",
        "T20",
        "T22",
    ]
    assert all(problem.name == key for key, problem in CATALOGUE.items())


def test_catalogue_is_read_only():
    with pytest.raises(TypeError):
        CATALOGUE["T99"] = CATALOGUE["T1"]


def test_every_problem_is_exported_at_the_top_level():
    assert set(CATALOGUE) <= set(maesolvers.__all__)
    assert set(maesolvers.catalogue.__all__) == {
        *CATALOGUE,
        "CATALOGUE",
        "SHARED_TEST_SET",
    }
    assert all(getattr(maesolvers, k) is p for k, p in CATALOGUE.items())


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_fields_reject_points_of_the_wrong_dimension(name):
    problem = CATALOGUE[name]
    fields = [problem.f, problem.exact]
    if isinstance(problem.boundary, Dirichlet):
        fields.append(problem.boundary.g)
    for field in filter(None, fields):
        for shape in [(3, problem.dim + 1), (problem.dim + 1,), (problem.dim, 3)]:
            if shape[-1] == problem.dim:
                continue
            with pytest.raises(ValueError, match="shape"):
                field(np.zeros(shape))


def test_shared_test_set_is_in_the_catalogue():
    assert SHARED_TEST_SET == ("T1", "T2", "T20", "T8", "T12", "T9", "T10")
    assert set(SHARED_TEST_SET) <= set(CATALOGUE)


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_every_problem_has_an_exact_solution_or_observables(name):
    problem = CATALOGUE[name]
    assert problem.exact is not None or problem.observables
    assert problem.note


@pytest.mark.parametrize(
    ("name", "dim"), [("T1", 2), ("T9", 3), ("T10", 4), ("T2", 2), ("T22", 3)]
)
def test_unit_box_problems_have_their_dimension(name, dim):
    assert CATALOGUE[name].domain == Box((0,) * dim, (1,) * dim)


# Shared test-set specification, Section 4: |error| <= 1e-10 relative to the
# printed value, absolute when the printed value is 0; T12 with c = 0.

SPEC_POINT_VALUES = [
    ("T1", (0.3, 0.7), 1.33642748803, 2.82194072059),
    ("T1", (0.55, 0.55), 1.35323767642, 2.93915979526),
    ("T1", (0.9, 0.1), 1.50681778511, 4.13230970431),
    ("T2", (0.3, 0.7), -1.19163752878, 0.991866693116),
    ("T2", (0.55, 0.55), -1.1811011811, 1.027736026),
    ("T2", (0.95, 0.95), -0.441588043316, 52.5969756739),
    ("T20", (0.55, 0.6), 0.0, 0.0),
    ("T20", (0.3, 0.7), 0.00343145750508, 0.292893218813),
    ("T20", (0.9, 0.1), 0.0668629150102, 0.646446609407),
    ("T8", (0.0, 0.0), -0.5, 1.0),
    ("T8", (0.3, -0.4), -0.375, 1.0),
    ("T8", (0.6, 0.6), -0.14, 1.0),
    ("T12", (0.0, 0.0), 0.0, 1.0),
    ("T12", (0.3, -0.4), 1.18, 1.0),
    ("T12", (0.6, 0.6), 2.55, 1.0),
    ("T9", (0.3, 0.7, 0.2), 1.36342511413, 4.10590486774),
    ("T9", (0.55, 0.55, 0.55), 1.57420439718, 7.44128451774),
    ("T9", (0.9, 0.1, 0.8), 2.07508060767, 21.9806242622),
    ("T10", (0.3, 0.7, 0.2, 0.4), 1.47698079388, 8.47070181635),
    ("T10", (0.55, 0.55, 0.55, 0.55), 1.83125220889, 24.8533490859),
    ("T10", (0.9, 0.1, 0.8, 0.3), 2.17059212718, 56.6047757677),
]


def _close(value, reference):
    tol = 1e-10 * abs(reference) if reference != 0.0 else 1e-10
    return abs(float(value) - reference) <= tol


@pytest.mark.parametrize(("name", "point", "u", "f"), SPEC_POINT_VALUES)
def test_spec_point_values(name, point, u, f):
    problem = CATALOGUE[name]
    x = np.array(point)
    assert _close(problem.exact(x), u)
    assert _close(problem.f(x), f)


def test_spec_covers_the_shared_test_set():
    assert {row[0] for row in SPEC_POINT_VALUES} == set(SHARED_TEST_SET)


# Hand-computed values of the catalogue-only problems.

_SQRT2 = np.sqrt(2.0)
_Q0 = 1.0 / (256.0 * np.pi**3) + 1.0 / (32.0 * np.pi)  # q(0) of T16

HAND_VALUES = [
    # |x| = 1: u = 2 sqrt 2 / 3, f = 1/|x| = 1.
    ("T3", (0.6, 0.8), 2.0 * _SQRT2 / 3.0, 1.0),
    # At the centre u is on its flat part 0.08 and f = 0; at (1, 1),
    # r^2 = 1/2 and r^2 - 0.16 = 0.34:
    # u = 0.08 + (sqrt(0.5 * 0.34) - 0.16 ln((sqrt 0.5 + sqrt 0.34) / 0.4)) / 2.
    ("T4", (0.5, 0.5), 0.08, 0.0),
    (
        "T4",
        (1.0, 1.0),
        0.08
        + 0.5 * (np.sqrt(0.17) - 0.16 * np.log((np.sqrt(0.5) + np.sqrt(0.34)) / 0.4)),
        1.0,
    ),
    # Either side of the circle of radius 0.4: r = 0.41, sqrt(r^2 - 0.16) = 0.09.
    ("T4", (0.5, 0.89), 0.08, 0.0),
    ("T4", (0.5, 0.91), 0.08 + 0.5 * (0.41 * 0.09 - 0.16 * np.log(1.25)), 1.0),
    # r = sqrt 2: sqrt(r^2 - 1) = 1, u = sqrt 2 - ln(sqrt 2 + 1); inside the
    # unit circle u = f = 0.
    ("T5", (1.0, 1.0), _SQRT2 - np.log(_SQRT2 + 1.0), 4.0),
    ("T5", (0.5, 0.5), 0.0, 0.0),
    # Either side of the unit circle: r^2 = 0.9684 and 1.0825.
    ("T5", (0.6, 0.78), 0.0, 0.0),
    (
        "T5",
        (0.6, 0.85),
        np.sqrt(1.0825 * 0.0825) - np.log(np.sqrt(1.0825) + np.sqrt(0.0825)),
        4.0,
    ),
    # At (1/2, 0) the exponential is 7/10: u = 0.7 + 0.25 - 0.5,
    # f = 0.49 * 2 + 4 * 0.7 = 3.78.
    ("T6", (0.5, 0.0), 0.45, 3.78),
    # q'(0) = 0, q''(0) = -2 pi, so u = 4 q(0)^2 and
    # f = 1 - 16 pi q(0) + 64 pi^2 q(0)^2 = (1 - 8 pi q(0))^2.
    ("T16", (0.0, 0.0), 4.0 * _Q0**2, (1.0 - 8.0 * np.pi * _Q0) ** 2),
    # At the origin u = -sqrt 3, f = 3 * 3^(-5/2).
    ("T22", (0.0, 0.0, 0.0), -np.sqrt(3.0), 3.0**-1.5),
]


@pytest.mark.parametrize(("name", "point", "u", "f"), HAND_VALUES)
def test_hand_computed_values(name, point, u, f):
    problem = CATALOGUE[name]
    x = np.array(point)
    assert problem.exact(x) == pytest.approx(u, rel=1e-12, abs=1e-15)
    assert problem.f(x) == pytest.approx(f, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    ("name", "point"), [("T2", (1.0, 1.0)), ("T3", (0.0, 0.0)), ("T22", (1.0,) * 3)]
)
def test_f_is_infinite_at_its_singular_point(name, point):
    problem = CATALOGUE[name]
    with np.errstate(all="raise"):
        assert problem.f(np.array(point)) == np.inf
        assert np.isfinite(problem.exact(np.array(point)))


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_f_and_exact_are_finite_on_the_shared_grid_away_from_singular_points(name):
    problem = CATALOGUE[name]
    grid = shared_grid(problem.domain)
    singular = {"T2": (1, 1), "T3": (0, 0), "T22": (1, 1, 1)}.get(name)
    if singular is not None:
        grid = grid[np.any(grid != singular, axis=-1)]
    with np.errstate(all="raise"):
        assert np.all(np.isfinite(problem.f(grid))) and np.all(problem.f(grid) >= 0)
        if problem.exact is not None:
            assert np.all(np.isfinite(problem.exact(grid)))


# det D^2 u = f by finite differences


def _hessian_det(u, p, h=1e-4):
    """det of the central-difference Hessian of ``u`` at points ``p`` (M, d)."""
    d = p.shape[-1]
    e = h * np.eye(d)
    hess = np.empty((len(p), d, d))
    for i in range(d):
        hess[:, i, i] = (u(p + e[i]) - 2.0 * u(p) + u(p - e[i])) / h**2
        for j in range(i):
            cross = (
                u(p + e[i] + e[j])
                - u(p + e[i] - e[j])
                - u(p - e[i] + e[j])
                + u(p - e[i] - e[j])
            ) / (4.0 * h**2)
            hess[:, i, j] = hess[:, j, i] = cross
    return np.linalg.det(hess)


# Sets where u is not smooth: the circles where D^2 u jumps.
KINKS = {
    "T4": ((0.5, 0.5), 0.4),
    "T5": ((0.0, 0.0), 1.0),
    "T20": ((0.5, 0.5), 0.2),
}


def _smooth_interior_points(problem, rng, count=300, margin=0.05):
    """Random points at least ``margin`` from the boundary and from kinks."""
    box = problem.domain.bbox
    d = problem.dim
    offsets = margin * np.concatenate([np.eye(d), -np.eye(d)])
    if d == 2:
        diagonals = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) / np.sqrt(2)
        offsets = np.concatenate([offsets, margin * diagonals])
    p = rng.uniform(box.lo, box.hi, size=(50 * count, d))
    keep = np.all([problem.domain.interior(p + o) for o in offsets], axis=0)
    if problem.name in KINKS:
        center, radius = KINKS[problem.name]
        r = np.linalg.norm(p - center, axis=-1)
        keep &= np.abs(r - radius) > margin
    p = p[keep]
    assert len(p) >= count
    return p[:count]


@pytest.mark.parametrize(
    "name", [k for k, p in CATALOGUE.items() if p.exact is not None]
)
def test_exact_solution_satisfies_the_equation(name):
    problem = CATALOGUE[name]
    p = _smooth_interior_points(problem, np.random.default_rng(0))
    assert np.allclose(
        _hessian_det(problem.exact, p), problem.f(p), rtol=1e-4, atol=1e-6
    )


# Dirichlet data is the trace of the exact solution


def _boundary_points(domain, rng, count=400):
    """Random points on the boundary of a catalogue domain."""
    if isinstance(domain, Box):
        d = domain.dim
        p = rng.uniform(domain.lo, domain.hi, size=(count, d))
        axis = rng.integers(d, size=count)
        side = np.where(rng.integers(2, size=(count, 1)) == 1, domain.hi, domain.lo)
        p[np.arange(count), axis] = side[np.arange(count), axis]
        return p
    if isinstance(domain, Disk):
        t = rng.uniform(0.0, 2.0 * np.pi, count)
        return np.asarray(domain.center) + domain.radius * np.stack(
            [np.cos(t), np.sin(t)], axis=-1
        )
    assert isinstance(domain, CutQuarterDisk)
    corner = np.array([(1.0 + np.sqrt(7.0)) / 4.0, (np.sqrt(7.0) - 1.0) / 4.0])
    s = rng.uniform(0.0, 1.0, count)[:, None]
    t = rng.uniform(np.arctan2(corner[1], corner[0]), np.pi / 2, count)
    return np.concatenate(
        [
            s * [0.0, 1.0],  # the side on the y axis
            s * [0.5, 0.0],  # the side on the x axis
            (1 - s) * [0.5, 0.0] + s * corner,  # the cut y = x - 1/2
            np.stack([np.cos(t), np.sin(t)], axis=-1),  # the arc
        ]
    )


def test_boundary_sample_lies_on_the_boundary():
    """Every catalogue domain is convex and contains the centre of its bbox, so
    a boundary point moves inside towards that centre and outside away from it.
    """
    rng = np.random.default_rng(1)
    for problem in CATALOGUE.values():
        domain = problem.domain
        b = _boundary_points(domain, rng)
        towards_centre = np.mean([domain.bbox.lo, domain.bbox.hi], axis=0) - b
        assert np.all(domain.interior(b + 1e-9 * towards_centre))
        assert not np.any(domain.contains(b - 1e-9 * towards_centre))


@pytest.mark.parametrize(
    "name", [k for k, p in CATALOGUE.items() if isinstance(p.boundary, Dirichlet)]
)
def test_dirichlet_data_is_the_trace_of_the_exact_solution(name):
    problem = CATALOGUE[name]
    b = _boundary_points(problem.domain, np.random.default_rng(2))
    if problem.exact is None:
        assert name == "T7"
        assert np.all(problem.boundary.g(b) == 1.0)
        return
    assert np.allclose(problem.boundary.g(b), problem.exact(b), rtol=1e-13, atol=1e-15)


# The area of grad u(Omega) is the integral of f


def _gradient(u, p, h=1e-6):
    e = h * np.eye(p.shape[-1])
    return np.stack([(u(p + ei) - u(p - ei)) / (2 * h) for ei in e], axis=-1)


def _box_loop(box, count):
    """``4 count`` points around the boundary of a 2D box, counterclockwise."""
    (x0, y0), (x1, y1) = box.lo, box.hi
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    t = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
    return np.concatenate(
        [a + t * (b - a) for a, b in zip(corners, np.roll(corners, -1, 0))]
    )


def _enclosed_area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# Integral of f over the domain, by hand. T1: with A = int_0^1 exp(t^2) dt,
# int (1 + x^2 + y^2) exp(x^2 + y^2) = A^2 + 2 A (e - A)/2 = A e. T3: the
# integral of 1/|x| over (0, a)^2 is 2 a ln(1 + sqrt 2). T4, T5: the area
# where f > 0 times f. T20: (1 - 0.04 pi) - 0.2 (4 ln(1 + sqrt 2) - 0.4 pi),
# the square of side 1 about the centre being four squares of side 1/2.
_LN = np.log(1.0 + np.sqrt(2.0))
INTEGRAL_OF_F = {
    "T1": np.e * np.sqrt(np.pi) / 2.0 * erfi(1.0),
    "T3": 2.0 * _LN,
    "T4": 1.0 - 0.16 * np.pi,
    "T5": 4.0 * (1.5**2 - np.pi / 4.0),
    "T20": 1.0 + 0.04 * np.pi - 0.8 * _LN,
}


@pytest.mark.parametrize("name", list(INTEGRAL_OF_F))
def test_gradient_image_has_the_mass_of_f(name):
    """For a C^1 convex u on a convex domain, grad u(Omega) is the region the
    image of the boundary encloses; its area is the Monge-Ampere measure of
    Omega, which equals the integral of f only if the measure has no singular
    part, as on a circle where grad u jumps."""
    problem = CATALOGUE[name]
    image = _gradient(problem.exact, _box_loop(problem.domain, 20000))
    assert _enclosed_area(image) == pytest.approx(INTEGRAL_OF_F[name], rel=1e-7)


# Transport problems


def test_t12_gradient_maps_the_circle_onto_the_ellipse():
    problem = CATALOGUE["T12"]
    b = _boundary_points(problem.domain, np.random.default_rng(3))
    grad = _gradient(problem.exact, b)
    target = problem.boundary.target
    level = np.sum(((grad - target.center) / target.semi_axes) ** 2, axis=-1)
    assert np.allclose(level, 1.0)


def test_t16_gradient_maps_each_side_of_the_square_into_itself():
    problem = CATALOGUE["T16"]
    b = _boundary_points(problem.domain, np.random.default_rng(4))
    grad = _gradient(problem.exact, b)
    on_side = np.isclose(np.abs(b), 0.5)
    assert np.allclose(grad[on_side], b[on_side], atol=1e-8)
    inside = shared_grid(problem.domain, size=101)
    assert np.all(np.abs(_gradient(problem.exact, inside)) <= 0.5 + 1e-8)


@pytest.mark.parametrize(
    "name", [k for k, p in CATALOGUE.items() if isinstance(p.boundary, Transport)]
)
def test_transport_mass_balance(name):
    """The integral of f over the domain equals the target's area."""
    problem = CATALOGUE[name]
    target = problem.boundary.target
    if isinstance(target, Box):
        area = np.prod(np.subtract(target.hi, target.lo))
    else:
        area = np.pi * np.prod(target.semi_axes)
    domain = problem.domain
    nodes, weights = np.polynomial.legendre.leggauss(200)
    if isinstance(domain, Box):
        (x0, y0), (x1, y1) = domain.lo, domain.hi
        x = x0 + (nodes + 1) * (x1 - x0) / 2
        y = y0 + (nodes + 1) * (y1 - y0) / 2
        w = np.outer(weights, weights) * (x1 - x0) * (y1 - y0) / 4
        p = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
        integral = np.sum(w * problem.f(p))
    else:
        assert isinstance(domain, Disk) and domain.center == (0.0, 0.0)
        r = (nodes + 1) / 2 * domain.radius
        t = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
        rr, tt = np.meshgrid(r, t, indexing="ij")
        p = np.stack([rr * np.cos(tt), rr * np.sin(tt)], axis=-1)
        w = (weights * domain.radius / 2 * r)[:, None] * (2 * np.pi / len(t))
        integral = np.sum(w * problem.f(p))
    assert integral == pytest.approx(area, rel=1e-10)


# Observables


def test_t7_reports_u_at_the_origin():
    problem = CATALOGUE["T7"]
    result = Result(problem, lambda x: 0.3 + np.sum(np.asarray(x) ** 2, axis=-1))
    assert evaluate(result) == {"u_origin": pytest.approx(0.3)}


# Every included solver on every catalogue problem: it either solves and
# evaluate reports finite numbers, or it refuses with UnsupportedProblemError.


@pytest.mark.parametrize("solver", [FDPoisson(n=8), FDNewton(n=8)], ids=repr)
@pytest.mark.parametrize("name", list(CATALOGUE))
def test_every_solver_solves_or_refuses_every_problem(solver, name):
    problem = CATALOGUE[name]
    if not solver.supports(problem):
        with pytest.raises(UnsupportedProblemError):
            solver.solve(problem)
        return
    metrics = evaluate(solver.solve(problem))
    assert metrics and all(np.isfinite(v) for v in metrics.values()), metrics
