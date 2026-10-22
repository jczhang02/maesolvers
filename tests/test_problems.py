"""Each catalogue problem's exact solution must satisfy its own equation."""

import numpy as np
import pytest

from maesolvers import CATALOGUE, Box, Dirichlet, Disk


def _boundary_points(domain, n):
    """``n`` points per side of a 2D box, or ``n`` points on a circle."""
    if isinstance(domain, Box):
        t = np.linspace(0.0, 1.0, n)
        (x0, y0), (x1, y1) = domain.lo, domain.hi
        xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        return np.concatenate(
            [
                np.stack([xs, np.full(n, y0)], axis=-1),
                np.stack([xs, np.full(n, y1)], axis=-1),
                np.stack([np.full(n, x0), ys], axis=-1),
                np.stack([np.full(n, x1), ys], axis=-1),
            ]
        )
    assert isinstance(domain, Disk)
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    return np.asarray(domain.center) + domain.radius * circle


def _interior_points(problem, rng, count=200, margin=0.05):
    box = problem.domain.bbox
    pts = []
    while len(pts) < count:
        p = rng.uniform(box.lo, box.hi, size=(4 * count, 2))
        lo, hi = np.asarray(box.lo) + margin, np.asarray(box.hi) - margin
        keep = problem.domain.contains(p) & np.all((p > lo) & (p < hi), axis=-1)
        if hasattr(problem.domain, "radius"):
            r = np.sqrt(np.sum((p - problem.domain.center) ** 2, axis=-1))
            keep &= r < problem.domain.radius - margin
        pts.extend(p[keep])
    return np.asarray(pts[:count])


def _hessian_det(u, p, h=1e-4):
    e1, e2 = np.array([h, 0.0]), np.array([0.0, h])
    uxx = (u(p + e1) - 2 * u(p) + u(p - e1)) / h**2
    uyy = (u(p + e2) - 2 * u(p) + u(p - e2)) / h**2
    uxy = (u(p + e1 + e2) - u(p + e1 - e2) - u(p - e1 + e2) + u(p - e1 - e2)) / (
        4 * h**2
    )
    return uxx * uyy - uxy**2


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_exact_solution_satisfies_equation(name):
    problem = CATALOGUE[name]
    rng = np.random.default_rng(0)
    p = _interior_points(problem, rng)
    det = _hessian_det(problem.exact, p)
    assert np.allclose(det, problem.f(p), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_dirichlet_data_is_the_trace(name):
    problem = CATALOGUE[name]
    if not isinstance(problem.boundary, Dirichlet):
        pytest.skip("transport problem")
    b = _boundary_points(problem.domain, 64)
    assert np.allclose(problem.boundary.g(b), problem.exact(b))


def test_p3_gradient_maps_disk_onto_ellipse():
    problem = CATALOGUE["P3"]
    b = _boundary_points(problem.domain, 360)
    grad = np.stack([2 * b[:, 0] + 3.5, 0.5 * b[:, 1]], axis=-1)
    c, (a1, a2) = problem.boundary.target.center, problem.boundary.target.semi_axes
    on_boundary = ((grad[:, 0] - c[0]) / a1) ** 2 + ((grad[:, 1] - c[1]) / a2) ** 2
    assert np.allclose(on_boundary, 1.0)
