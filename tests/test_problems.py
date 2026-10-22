"""Each catalogue problem's exact solution must satisfy its own equation."""

import numpy as np
import pytest

from maesolvers import CATALOGUE


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
    if problem.boundary != "dirichlet":
        pytest.skip("transport problem")
    b = problem.domain.boundary_points(64)
    assert np.allclose(problem.g(b), problem.exact(b))


def test_p3_gradient_maps_disk_onto_ellipse():
    problem = CATALOGUE["P3"]
    b = problem.domain.boundary_points(360)
    grad = np.stack([2 * b[:, 0] + 3.5, 0.5 * b[:, 1]], axis=-1)
    c, (a1, a2) = problem.target.center, problem.target.semi_axes
    on_boundary = ((grad[:, 0] - c[0]) / a1) ** 2 + ((grad[:, 1] - c[1]) / a2) ** 2
    assert np.allclose(on_boundary, 1.0)
