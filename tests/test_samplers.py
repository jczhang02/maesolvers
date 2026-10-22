"""Collocation points: the sampler part of a neural solver, numpy only."""

import numpy as np
import pytest

from maesolvers import T1, T6, T7, T8, T12, Box, Dirichlet, Problem, UniformSampler
from maesolvers.core import CutQuarterDisk, Ellipse
from maesolvers.neural import Collocation, sample_interior


def test_interior_points_are_uniform_in_the_open_domain_and_reproducible():
    domain = CutQuarterDisk()
    points = sample_interior(domain, 2000, np.random.default_rng(1))
    assert points.shape == (2000, 2)
    assert np.all(domain.interior(points))
    again = sample_interior(domain, 2000, np.random.default_rng(1))
    np.testing.assert_array_equal(points, again)
    # Uniform: as many points left of x = 1/2 as a large uniform sample has.
    box = domain.bbox
    reference = np.random.default_rng(2).uniform(box.lo, box.hi, (10**5, 2))
    reference = reference[domain.interior(reference)]
    expected = np.mean(reference[:, 0] < 0.5)
    assert np.mean(points[:, 0] < 0.5) == pytest.approx(expected, abs=0.03)


class Circle:
    """The unit circle as a domain: a bounding box, but no interior."""

    dim = 2
    bbox = Box((-1, -1), (1, 1))

    def contains(self, x):
        return np.isclose(np.sum(np.square(x), axis=-1), 1.0)

    def interior(self, x):
        return np.zeros(np.shape(x)[:-1], dtype=bool)


def test_rejection_sampling_gives_up_on_a_domain_without_interior():
    with pytest.raises(ValueError, match=r"rejection sampling found 0 of 5 .* Circle"):
        sample_interior(Circle(), 5, np.random.default_rng(0))


def test_the_sampler_draws_interior_then_boundary_points_from_the_generator():
    sampler = UniformSampler(interior=300, boundary=40)
    points = sampler.sample(T6, np.random.default_rng(7))
    assert points.interior.shape == (300, 2) and points.boundary.shape == (40, 2)
    assert np.all(T6.domain.interior(points.interior))
    np.testing.assert_allclose(
        T6.domain.boundary_distance(points.boundary), 0, atol=1e-12
    )
    rng = np.random.default_rng(7)
    np.testing.assert_array_equal(points.interior, sample_interior(T6.domain, 300, rng))
    np.testing.assert_array_equal(points.boundary, T6.domain.sample_boundary(40, rng))


def test_the_sampler_needs_boundary_sampling_only_for_boundary_points():
    assert UniformSampler().supports(T6) and UniformSampler().supports(T8)
    assert UniformSampler().supports(T12)  # any boundary condition: a disk samples
    ellipse = Problem("E", Ellipse((0, 0), (2, 1)), T1.f, Dirichlet(T1.f))
    assert not UniformSampler().supports(ellipse)
    assert UniformSampler(boundary=0).supports(ellipse)
    points = UniformSampler(interior=5, boundary=0).sample(
        ellipse, np.random.default_rng(0)
    )
    assert points.boundary.shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"interior": 0}, ValueError, "interior must be at least 1"),
        ({"boundary": -1}, ValueError, "boundary must be at least 0"),
        ({"interior": 10.5}, TypeError, "interior must be an integer"),
    ],
)
def test_the_sampler_checks_its_counts(kwargs, error, message):
    with pytest.raises(error, match=message):
        UniformSampler(**kwargs)


def test_collocation_stores_read_only_float_copies():
    interior = np.array([[0.5, 0.5]])
    points = Collocation(interior=interior, boundary=np.zeros((0, 2)))
    interior[0, 0] = 9.0
    assert points.interior[0, 0] == 0.5 and points.dim == 2
    with pytest.raises(ValueError, match="read-only"):
        points.interior[0, 0] = 1.0


@pytest.mark.parametrize(
    ("interior", "boundary", "message"),
    [
        (np.zeros((0, 2)), np.zeros((1, 2)), r"interior must have shape \(M, d\)"),
        (np.zeros(2), np.zeros((1, 2)), r"interior must have shape \(M, d\)"),
        (np.zeros((3, 2)), np.zeros((1, 3)), r"boundary must have shape \(K, 2\)"),
    ],
)
def test_collocation_checks_its_shapes(interior, boundary, message):
    with pytest.raises(ValueError, match=message):
        Collocation(interior=interior, boundary=boundary)


def test_the_fingerprint_tells_point_sets_apart():
    sampler = UniformSampler(interior=50, boundary=10)
    first = sampler.sample(T1, np.random.default_rng(0)).fingerprint()
    assert first == sampler.sample(T1, np.random.default_rng(0)).fingerprint()
    assert first != sampler.sample(T1, np.random.default_rng(1)).fingerprint()
    assert len(first) == 64


def test_a_box_problem_in_any_dimension_samples():
    cube = Problem("C", Box((0,) * 3, (1,) * 3), T1.f, Dirichlet(T1.f))
    points = UniformSampler(interior=20, boundary=30).sample(
        cube, np.random.default_rng(0)
    )
    assert points.interior.shape == (20, 3) and points.boundary.shape == (30, 3)
    assert UniformSampler().supports(T7)
