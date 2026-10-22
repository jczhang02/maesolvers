"""Domains: validation, membership, bounding boxes, boundary points."""

import numpy as np
import pytest

from maesolvers.core import Box, Disk, Domain, Ellipse


@pytest.mark.parametrize(
    "domain",
    [
        Box((0, 0), (1, 1)),
        Box((0,) * 3, (1,) * 3),
        Disk((0, 0), 1),
        Ellipse((0, 0), (2, 1)),
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


def test_box_boundary_points_lie_on_the_sides_without_repeats():
    box = Box((0, -1), (2, 1))
    b = box.boundary_points(8)
    assert b.shape == (32, 2)
    assert np.all(box.contains(b)) and not np.any(box.interior(b))
    assert len(np.unique(b, axis=0)) == 32


def test_box_boundary_points_are_2d_only():
    with pytest.raises(ValueError, match="2D"):
        Box((0,) * 3, (1,) * 3).boundary_points(4)


def test_disk_membership_bbox_and_boundary():
    disk = Disk((1, 0), 2)
    assert disk.dim == 2
    assert disk.bbox == Box((-1, -2), (3, 2))
    points = np.array([[1.0, 0.0], [3.0, 0.0], [3.1, 0.0]])
    assert disk.contains(points).tolist() == [True, True, False]
    assert disk.interior(points).tolist() == [True, False, False]
    b = disk.boundary_points(16)
    assert np.allclose(np.linalg.norm(b - disk.center, axis=-1), 2.0)


@pytest.mark.parametrize(
    ("center", "radius", "message"),
    [((0, 0), 0.0, "positive"), ((0, 0), -1.0, "positive"), ((0, 0, 0), 1.0, "2D")],
)
def test_disk_rejects_bad_parameters(center, radius, message):
    with pytest.raises(ValueError, match=message):
        Disk(center, radius)


def test_ellipse_membership_bbox_and_boundary():
    ellipse = Ellipse((3.5, 0), (2, 0.5))
    assert ellipse.bbox == Box((1.5, -0.5), (5.5, 0.5))
    b = ellipse.boundary_points(32)
    level = ((b[:, 0] - 3.5) / 2) ** 2 + (b[:, 1] / 0.5) ** 2
    assert np.allclose(level, 1.0)
    assert ellipse.contains(np.array([5.5, 0.0])) and not ellipse.interior(
        np.array([5.5, 0.0])
    )


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
