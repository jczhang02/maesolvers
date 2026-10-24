"""Collocation points: the sampler part of a neural solver, numpy only."""

import hashlib

import numpy as np
import pytest

from maesolvers import (
    T1,
    T6,
    T7,
    T8,
    T12,
    T16,
    Box,
    Dirichlet,
    Disk,
    Problem,
    Transport,
    TransportSampler,
    UniformSampler,
)
from maesolvers.core import CutQuarterDisk
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


class BareDisk:
    """The unit disk as a bare Domain, without boundary sampling."""

    dim = 2
    bbox = Box((-1, -1), (1, 1))

    def contains(self, x):
        return np.sum(np.square(x), axis=-1) <= 1.0

    def interior(self, x):
        return np.sum(np.square(x), axis=-1) < 1.0


def test_rejection_sampling_gives_up_on_a_domain_without_interior():
    with pytest.raises(ValueError, match=r"rejection sampling found 0 of 5 .* Circle"):
        sample_interior(Circle(), 5, np.random.default_rng(0))


def test_the_sampler_draws_interior_then_boundary_points_from_the_generator():
    sampler = UniformSampler(interior_points=300, boundary_points=40)
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
    bare = Problem("E", BareDisk(), T1.f, Dirichlet(T1.f))
    assert not UniformSampler().supports(bare)
    assert UniformSampler(boundary_points=0).supports(bare)
    points = UniformSampler(interior_points=5, boundary_points=0).sample(
        bare, np.random.default_rng(0)
    )
    assert points.boundary.shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"interior_points": 0}, ValueError, "interior_points must be at least 1"),
        ({"boundary_points": -1}, ValueError, "boundary_points must be at least 0"),
        ({"interior_points": 10.5}, TypeError, "interior_points must be an integer"),
    ],
)
def test_the_sampler_checks_its_counts(kwargs, error, message):
    with pytest.raises(error, match=message):
        UniformSampler(**kwargs)


def test_collocation_stores_read_only_float_copies():
    interior = np.array([[0.5, 0.5]])
    points = Collocation(interior=interior, boundary=np.zeros((0, 2)))
    interior[0, 0] = 9.0
    assert points.interior[0, 0] == 0.5
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


def test_collocation_has_no_target_points_unless_given():
    points = Collocation(interior=np.zeros((3, 2)), boundary=np.zeros((1, 2)))
    assert points.target_boundary.shape == (0, 2)
    with pytest.raises(ValueError, match=r"target_boundary must have shape \(K, 2\)"):
        Collocation(
            interior=np.zeros((3, 2)),
            boundary=np.zeros((1, 2)),
            target_boundary=np.zeros((4, 3)),
        )
    target = Collocation(
        interior=np.zeros((3, 2)),
        boundary=np.zeros((1, 2)),
        target_boundary=np.ones((4, 2)),
    )
    with pytest.raises(ValueError, match="read-only"):
        target.target_boundary[0, 0] = 1.0


def test_the_fingerprint_of_a_set_without_target_points_is_unchanged():
    points = UniformSampler(interior_points=5, boundary_points=3).sample(
        T1, np.random.default_rng(0)
    )
    digest = hashlib.sha256(points.interior.astype("<f8").tobytes())
    digest.update(points.boundary.astype("<f8").tobytes())
    assert points.fingerprint() == digest.hexdigest()
    with_target = Collocation(
        interior=points.interior,
        boundary=points.boundary,
        target_boundary=np.ones((2, 2)),
    )
    assert with_target.fingerprint() != points.fingerprint()


def test_the_fingerprint_tells_point_sets_apart():
    sampler = UniformSampler(interior_points=50, boundary_points=10)
    first = sampler.sample(T1, np.random.default_rng(0)).fingerprint()
    assert first == sampler.sample(T1, np.random.default_rng(0)).fingerprint()
    assert first != sampler.sample(T1, np.random.default_rng(1)).fingerprint()
    assert len(first) == 64


def test_a_box_problem_in_any_dimension_samples():
    cube = Problem("C", Box((0,) * 3, (1,) * 3), T1.f, Dirichlet(T1.f))
    points = UniformSampler(interior_points=20, boundary_points=30).sample(
        cube, np.random.default_rng(0)
    )
    assert points.interior.shape == (20, 3) and points.boundary.shape == (30, 3)
    assert UniformSampler().supports(T7)


def test_the_transport_sampler_draws_the_uniform_points_then_the_target_points():
    sampler = TransportSampler(
        interior_points=300, boundary_points=40, target_points=50
    )
    points = sampler.sample(T12, np.random.default_rng(7))
    uniform = UniformSampler(interior_points=300, boundary_points=40)
    rng = np.random.default_rng(7)
    first = uniform.sample(T12, rng)
    np.testing.assert_array_equal(points.interior, first.interior)
    np.testing.assert_array_equal(points.boundary, first.boundary)
    target = T12.boundary.target
    np.testing.assert_array_equal(
        points.target_boundary, target.sample_boundary(50, rng)
    )
    level = np.sum(((points.target_boundary - (3.5, 0)) / (2, 0.5)) ** 2, axis=-1)
    np.testing.assert_allclose(level, 1.0, rtol=1e-12)
    assert TransportSampler().sample(
        T12, np.random.default_rng(0)
    ).target_boundary.shape == (
        800,
        2,
    )


def test_the_transport_sampler_needs_a_transport_problem_with_sampled_boundaries():
    assert TransportSampler().supports(T12) and TransportSampler().supports(T16)
    assert not TransportSampler().supports(T1)  # Dirichlet
    bare = Problem("B", BareDisk(), T12.f, T12.boundary)
    assert not TransportSampler().supports(bare)
    onto_bare = Problem("O", Disk((0, 0), 1), T12.f, Transport(BareDisk()))
    assert not TransportSampler().supports(onto_bare)


@pytest.mark.parametrize(
    "name", ["interior_points", "boundary_points", "target_points"]
)
def test_the_transport_sampler_checks_its_counts(name):
    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        TransportSampler(**{name: 0})
