"""Domains: validation, membership, bounding boxes."""

import numpy as np
import pytest

from maesolvers.core import Box, CutQuarterDisk, Disk, Domain, Ellipse


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
    assert domain.interior(corner - [1e-9, 0.0])
    assert not domain.contains(corner + [1e-9, 0.0])


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
