"""Domains: validation, membership, bounding boxes, boundary sampling and distance."""

import numpy as np
import pytest

from maesolvers.core import (
    BoundaryDistance,
    BoundarySampling,
    Box,
    CutQuarterDisk,
    Disk,
    Domain,
    Ellipse,
)
from maesolvers.core.domains import _allocate


@pytest.mark.parametrize(
    "domain",
    [
        Box((0, 0), (1, 1)),
        Box((0,) * 3, (1,) * 3),
        Disk((0, 0), 1),
        Ellipse((0, 0), (2, 1)),
        CutQuarterDisk(),
    ],
)
def test_every_domain_satisfies_the_protocol(domain):
    assert isinstance(domain, Domain)


def test_box_is_d_dimensional_and_stores_floats():
    box = Box((0, 0, 0, 0), (1, 2, 3, 4))
    assert box.dim == 4
    assert box.lo == (0.0, 0.0, 0.0, 0.0)
    assert all(isinstance(v, float) for v in box.hi)
    assert box.bbox is box


@pytest.mark.parametrize(
    ("lo", "hi", "message"),
    [
        ((0, 0), (1,), "same length"),
        ((), (), "same length"),
        ((0, 1), (1, 1), "lo < hi"),
        ((0, np.inf), (1, 2), "finite"),
    ],
)
def test_box_rejects_bad_corners(lo, hi, message):
    with pytest.raises(ValueError, match=message):
        Box(lo, hi)


def test_box_rejects_non_numbers():
    with pytest.raises(TypeError, match="sequence of numbers"):
        Box(0.0, 1.0)


def test_box_membership_closed_and_open():
    box = Box((0, 0, 0), (1, 1, 1))
    points = np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 0.0], [1.5, 0.5, 0.5]])
    assert box.contains(points).tolist() == [True, True, False]
    assert box.interior(points).tolist() == [True, False, False]


def test_membership_keeps_the_leading_shape():
    box = Box((0, 0), (1, 1))
    assert box.contains(np.zeros((3, 4, 2))).shape == (3, 4)


def test_membership_rejects_points_of_the_wrong_dimension():
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        Box((0, 0), (1, 1)).contains(np.zeros((5, 3)))


def test_disk_membership_and_bbox():
    disk = Disk((1, 0), 2)
    assert disk.dim == 2
    assert disk.bbox == Box((-1, -2), (3, 2))
    points = np.array([[1.0, 0.0], [3.0, 0.0], [3.1, 0.0]])
    assert disk.contains(points).tolist() == [True, True, False]
    assert disk.interior(points).tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("center", "radius", "message"),
    [((0, 0), 0.0, "positive"), ((0, 0), -1.0, "positive"), ((0, 0, 0), 1.0, "2D")],
)
def test_disk_rejects_bad_parameters(center, radius, message):
    with pytest.raises(ValueError, match=message):
        Disk(center, radius)


def test_ellipse_membership_and_bbox():
    ellipse = Ellipse((3.5, 0), (2, 0.5))
    assert ellipse.bbox == Box((1.5, -0.5), (5.5, 0.5))
    assert ellipse.contains(np.array([5.5, 0.0])) and not ellipse.interior(
        np.array([5.5, 0.0])
    )


def test_cut_quarter_disk_bbox_touches_the_domain_on_every_side():
    domain = CutQuarterDisk()
    root7 = np.sqrt(7.0)
    assert domain.dim == 2
    assert domain.bbox == Box((0, 0), ((1 + root7) / 4, 1))
    # (1/2, 0), where the cut meets the x axis, and (0, 1) lie on the
    # boundary; the corner where the cut meets the circle, which fixes the
    # right side of the bbox, does too up to rounding.
    edge = np.array([[0.5, 0.0], [0.0, 1.0]])
    assert np.all(domain.contains(edge)) and not np.any(domain.interior(edge))
    corner = np.array([(1 + root7) / 4, (root7 - 1) / 4])
    assert domain.interior(corner - np.array([1e-9, 0.0]))
    assert not domain.contains(corner + np.array([1e-9, 0.0]))


@pytest.mark.parametrize(
    ("point", "inside"),
    [
        ((0.3, 0.3), True),
        ((0.8, 0.2), False),  # below the cut y = x - 1/2
        ((0.7, 0.75), False),  # outside the unit circle
        ((-0.1, 0.5), False),
        ((0.3, -0.1), False),
    ],
)
def test_cut_quarter_disk_membership(point, inside):
    domain = CutQuarterDisk()
    assert domain.contains(np.array(point)) == inside
    assert domain.interior(np.array(point)) == inside


@pytest.mark.parametrize(
    ("center", "axes", "message"),
    [((0, 0), (1, 0), "positive"), ((0, 0), (1,), "2D"), ((0,), (1, 1), "2D")],
)
def test_ellipse_rejects_bad_parameters(center, axes, message):
    with pytest.raises(ValueError, match=message):
        Ellipse(center, axes)


def test_domains_are_frozen_and_hashable():
    box = Box((0, 0), (1, 1))
    with pytest.raises(AttributeError):
        box.lo = (1, 1)
    assert len({box, Box((0.0, 0.0), (1.0, 1.0)), Disk((0, 0), 1)}) == 2


SAMPLED = [
    Box((0, 0), (1, 1)),
    Box((0, -1, 2), (1, 2, 2.5)),
    Box((-1,), (3,)),
    Disk((1, -1), 2),
    CutQuarterDisk(),
]


def _on_the_boundary(domain, points):
    """Whether every point is in the closed domain up to rounding, at distance 0."""
    near = domain.contains(points) | (domain.boundary_distance(points) > -1e-12)
    return bool(np.all(near) and np.allclose(domain.boundary_distance(points), 0.0))


@pytest.mark.parametrize("domain", SAMPLED)
def test_sampled_boundary_points_lie_on_the_boundary(domain):
    assert isinstance(domain, BoundarySampling)
    assert isinstance(domain, BoundaryDistance)
    points = domain.sample_boundary(500, np.random.default_rng(0))
    assert points.shape == (500, domain.dim)
    assert _on_the_boundary(domain, points)


@pytest.mark.parametrize("domain", SAMPLED)
def test_boundary_points_depend_on_the_generator_alone(domain):
    first = domain.sample_boundary(50, np.random.default_rng(3))
    again = domain.sample_boundary(50, np.random.default_rng(3))
    other = domain.sample_boundary(50, np.random.default_rng(4))
    np.testing.assert_array_equal(first, again)
    assert np.array_equal(first, other) == (domain.dim == 1)  # 1D: two end points


@pytest.mark.parametrize("domain", SAMPLED)
def test_no_boundary_points_is_an_empty_array(domain):
    assert domain.sample_boundary(0, np.random.default_rng(0)).shape == (0, domain.dim)


def test_the_number_of_boundary_points_is_checked():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be at least 0"):
        Box((0, 0), (1, 1)).sample_boundary(-1, rng)
    with pytest.raises(TypeError, match="n must be an integer"):
        CutQuarterDisk().sample_boundary(2.5, rng)


def test_box_faces_get_points_in_proportion_to_their_area():
    box = Box((0, 0, 0), (1, 2, 3))  # face areas 6, 6, 3, 3, 2, 2 for x, y, z
    points = box.sample_boundary(220, np.random.default_rng(0))
    on_face = [
        np.sum(points[:, axis] == end)
        for axis in range(3)
        for end in (box.lo[axis], box.hi[axis])
    ]
    assert on_face == [60, 60, 30, 30, 20, 20]


def test_cut_quarter_disk_sides_get_points_in_proportion_to_their_length():
    domain = CutQuarterDisk()
    points = domain.sample_boundary(1000, np.random.default_rng(0))
    x, y = points[:, 0], points[:, 1]
    bottom, left = np.sum(y == 0.0), np.sum(x == 0.0)
    arc = np.sum(np.isclose(x**2 + y**2, 1.0) & (x > 0) & (y > 0))
    cut = np.sum(np.isclose(y, x - 0.5) & (y > 0))
    cut_length = (np.sqrt(7.0) - 1.0) / (2.0 * np.sqrt(2.0))
    corner_angle = np.arctan2((np.sqrt(7.0) - 1.0) / 4.0, (1.0 + np.sqrt(7.0)) / 4.0)
    lengths = np.array([0.5, cut_length, np.pi / 2 - corner_angle, 1.0])
    expected = 1000 * lengths / lengths.sum()
    assert (bottom, cut, arc, left) == tuple(int(v) for v in _allocate(1000, lengths))
    np.testing.assert_allclose([bottom, cut, arc, left], expected, atol=1.0)


def test_allocation_adds_up_and_rounds_by_largest_remainder():
    counts = _allocate(10, np.array([1.0, 1.0, 1.0]))
    assert counts.tolist() == [4, 3, 3]
    assert _allocate(0, np.array([2.0, 1.0])).tolist() == [0, 0]
    assert _allocate(7, np.array([0.5, 0.25, 0.25])).sum() == 7


@pytest.mark.parametrize(
    ("domain", "point", "distance"),
    [
        (Box((0, 0), (1, 2)), (0.25, 1.5), 0.25),
        (Box((0, 0, 0), (1, 1, 1)), (0.5, 0.5, 0.9), 0.1),
        (Disk((1, 0), 2), (0.5, 0.0), 1.5),
        (CutQuarterDisk(), (0.1, 0.5), 0.1),  # the y axis is nearest
        (CutQuarterDisk(), (0.5, 0.1), 0.1 / np.sqrt(2.0)),  # the cut
        (CutQuarterDisk(), (0.3, 0.8), 1.0 - np.hypot(0.3, 0.8)),  # the arc
    ],
)
def test_boundary_distance_is_the_distance_to_the_nearest_side(domain, point, distance):
    assert domain.boundary_distance(np.array(point)) == pytest.approx(distance)


def test_boundary_distance_keeps_the_leading_shape():
    assert CutQuarterDisk().boundary_distance(np.full((3, 4, 2), 0.2)).shape == (3, 4)


def test_ellipse_neither_samples_its_boundary_nor_knows_its_distance():
    ellipse = Ellipse((0, 0), (2, 1))
    assert not isinstance(ellipse, BoundarySampling)
    assert not isinstance(ellipse, BoundaryDistance)
