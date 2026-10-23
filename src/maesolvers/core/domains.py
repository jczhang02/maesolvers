"""Computational domains and transport targets.

Points are arrays of shape ``(..., d)`` and predicates return boolean arrays of
shape ``(...)``. Every domain is a frozen dataclass that satisfies the
:class:`Domain` protocol. :class:`Box`, :class:`Disk` and
:class:`CutQuarterDisk` also satisfy :class:`BoundarySampling` and
:class:`BoundaryDistance`, the capabilities the neural solvers need;
:class:`Ellipse`, a transport target, satisfies :class:`BoundarySampling`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from maesolvers.core._params import floats, integer, points

Field = Callable[[np.ndarray], np.ndarray]
"""Maps points of shape ``(..., d)`` to values of shape ``(...)``."""

HessianField = Callable[[np.ndarray], np.ndarray]
"""Maps points of shape ``(..., d)`` to matrices of shape ``(..., d, d)``."""


@runtime_checkable
class Domain(Protocol):
    """What metrics and solvers need from a domain.

    The protocol is closed at 0.1: a user's own domain needs these four
    members and no more. Further capabilities (sampling, a signed distance,
    a mesh) come as separate protocols, which a solver requires through
    :meth:`~maesolvers.core.solver.Solver.supports`.
    """

    @property
    def dim(self) -> int:
        """Space dimension ``d``."""
        ...

    @property
    def bbox(self) -> Box:
        """Smallest axis-aligned box that contains the domain."""
        ...

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed domain."""
        ...

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open domain."""
        ...


@runtime_checkable
class BoundarySampling(Protocol):
    """A domain that draws random points on its boundary.

    A capability beside :class:`Domain` (ADR 0007): a solver that needs it,
    such as a neural solver's sampler, asks for it in its ``supports``.
    """

    def sample_boundary(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points on the boundary, uniformly with respect to its area.

        Parameters
        ----------
        n
            Number of points, at least 0.
        rng
            The random generator; the points depend on it alone.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, d)``. Each piece of the boundary gets a number of
            points proportional to its area (its length in 2D), rounded so
            that the numbers add up to ``n``, and the points on a piece are
            uniform on it.
        """
        ...


@runtime_checkable
class BoundaryDistance(Protocol):
    """A domain that knows the distance from its points to its boundary.

    A capability beside :class:`Domain` (ADR 0007), asked for in a solver's
    ``supports``.
    """

    def boundary_distance(self, x: np.ndarray) -> np.ndarray:
        """Distance from each point of the closed domain to the boundary.

        Parameters
        ----------
        x
            Points of shape ``(..., d)`` in the closed domain.

        Returns
        -------
        numpy.ndarray
            Shape ``(...)``: zero on the boundary, positive inside. For points
            outside the domain the value has no meaning.
        """
        ...


@runtime_checkable
class SupportFunction(Protocol):
    """A convex transport target that knows its own support function.

    A capability beside :class:`Domain` (ADR 0007) for a target set that
    :func:`~maesolvers.core.transport.support_function` does not know, such
    as a rotated ellipse or a polygon: that function calls this method for
    such a target, so a solver that represents the target by its support
    function, such as :class:`~maesolvers.classical.SecondBoundary`, accepts
    it. :class:`Box`, :class:`Disk` and :class:`Ellipse` need no method;
    :func:`~maesolvers.core.transport.support_function` knows them.
    """

    def support_function(self, n: np.ndarray) -> np.ndarray:
        """Return ``sup_{y in Y} y . n`` for each direction.

        Parameters
        ----------
        n
            Directions of shape ``(..., d)``, not necessarily unit vectors.

        Returns
        -------
        numpy.ndarray
            Shape ``(...)``, positively homogeneous in ``n``.
        """
        ...


def _allocate(n: int, sizes: np.ndarray) -> np.ndarray:
    """Split ``n`` points among pieces in proportion to their sizes.

    Largest remainder: each piece gets the floor of its share, and the points
    left over go to the pieces with the largest fractional parts, the first
    piece first on a tie.

    Parameters
    ----------
    n
        Number of points, at least 0.
    sizes
        Shape ``(k,)``, positive: the length or area of each piece.

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)``, integers that add up to ``n``.
    """
    share = n * sizes / np.sum(sizes)
    counts = np.floor(share).astype(int)
    order = np.argsort(-(share - counts), kind="stable")
    counts[order[: n - int(np.sum(counts))]] += 1
    return counts


@dataclass(frozen=True)
class Box:
    """Axis-aligned box ``[lo[0], hi[0]] x ... x [lo[d-1], hi[d-1]]``.

    Parameters
    ----------
    lo, hi
        Lower and upper corners, tuples of the same length ``d >= 1`` with
        ``lo[i] < hi[i]``.
    """

    lo: tuple[float, ...]
    hi: tuple[float, ...]

    def __post_init__(self) -> None:
        lo, hi = floats("Box.lo", self.lo), floats("Box.hi", self.hi)
        if len(lo) != len(hi):
            raise ValueError(
                f"Box.lo and Box.hi need the same length, got {lo} and {hi}"
            )
        if not all(a < b for a, b in zip(lo, hi, strict=True)):
            raise ValueError(f"Box needs lo < hi in every coordinate, got {lo}, {hi}")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def dim(self) -> int:
        """Space dimension ``d``, the length of ``lo``."""
        return len(self.lo)

    @property
    def bbox(self) -> Box:
        """The box itself."""
        return self

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed box."""
        x = points(x, self.dim)
        return np.all((x >= self.lo) & (x <= self.hi), axis=-1)

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open box."""
        x = points(x, self.dim)
        return np.all((x > self.lo) & (x < self.hi), axis=-1)

    def sample_boundary(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points on the faces, in proportion to their areas.

        The faces are taken in the order ``x_0 = lo[0]``, ``x_0 = hi[0]``,
        ``x_1 = lo[1]``, and so on; in 1D each face is a single point.

        Parameters
        ----------
        n
            Number of points, at least 0.
        rng
            The random generator.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, d)``.
        """
        n = integer("n", n, 0)
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        sides = hi - lo
        areas = np.repeat([np.prod(np.delete(sides, i)) for i in range(self.dim)], 2)
        pieces = []
        for face, count in enumerate(_allocate(n, areas)):
            axis, end = divmod(face, 2)
            piece = rng.uniform(lo, hi, size=(count, self.dim))
            piece[:, axis] = hi[axis] if end else lo[axis]
            pieces.append(piece)
        return np.concatenate(pieces)

    def boundary_distance(self, x: np.ndarray) -> np.ndarray:
        """Distance to the nearest face, for points in the closed box."""
        x = points(x, self.dim)
        return np.min(np.minimum(x - self.lo, np.asarray(self.hi) - x), axis=-1)


@dataclass(frozen=True)
class Ellipse:
    """Axis-aligned ellipse in the plane.

    Parameters
    ----------
    center
        Center ``(c1, c2)``.
    semi_axes
        Semi-axes ``(a1, a2)`` along the coordinate axes, both positive.
    """

    center: tuple[float, float]
    semi_axes: tuple[float, float]

    def __post_init__(self) -> None:
        center = floats("Ellipse.center", self.center)
        axes = floats("Ellipse.semi_axes", self.semi_axes)
        if len(center) != 2 or len(axes) != 2:
            raise ValueError(
                f"Ellipse is 2D: center and semi_axes need length 2, got {center}, {axes}"
            )
        if min(axes) <= 0.0:
            raise ValueError(f"Ellipse needs positive semi-axes, got {axes}")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "semi_axes", axes)

    @property
    def dim(self) -> int:
        """Space dimension, always 2."""
        return 2

    @property
    def bbox(self) -> Box:
        """The box ``center -/+ semi_axes``."""
        c, a = np.asarray(self.center), np.asarray(self.semi_axes)
        return Box(tuple(c - a), tuple(c + a))

    def _level(self, x: np.ndarray) -> np.ndarray:
        x = points(x, 2)
        return np.sum(((x - self.center) / np.asarray(self.semi_axes)) ** 2, axis=-1)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed ellipse."""
        return self._level(x) <= 1.0

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open ellipse."""
        return self._level(x) < 1.0

    def sample_boundary(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points on the ellipse, uniform in arc length.

        The point at parameter ``t`` is ``center + (a1 cos t, a2 sin t)``,
        and its speed ``|dp/dt|`` varies along the ellipse, so ``t`` is drawn
        uniform and kept with probability ``speed(t) / max speed``
        (rejection), which makes the points uniform in arc length, not in
        ``t``.

        Parameters
        ----------
        n
            Number of points, at least 0.
        rng
            The random generator.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)``, in the order they were accepted.
        """
        n = integer("n", n, 0)
        a1, a2 = self.semi_axes
        kept: list[np.ndarray] = []
        count = 0
        while count < n:
            t = rng.uniform(0.0, 2.0 * np.pi, size=2 * (n - count))
            speed = np.hypot(a1 * np.sin(t), a2 * np.cos(t))
            t = t[rng.uniform(0.0, max(a1, a2), size=t.size) < speed]
            kept.append(t)
            count += t.size
        t = np.concatenate(kept)[:n] if kept else np.zeros(0)
        circle = np.stack([a1 * np.cos(t), a2 * np.sin(t)], axis=-1)
        return np.asarray(self.center) + circle


@dataclass(frozen=True)
class Disk:
    """Disk in the plane.

    Parameters
    ----------
    center
        Center ``(c1, c2)``.
    radius
        Positive radius.
    """

    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        center = floats("Disk.center", self.center)
        if len(center) != 2:
            raise ValueError(f"Disk is 2D: center needs length 2, got {center}")
        radius = float(self.radius)
        if not radius > 0.0 or not np.isfinite(radius):
            raise ValueError(f"Disk needs a positive finite radius, got {self.radius}")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    @property
    def dim(self) -> int:
        """Space dimension, always 2."""
        return 2

    @property
    def bbox(self) -> Box:
        """The box ``center -/+ radius``."""
        c = np.asarray(self.center)
        return Box(tuple(c - self.radius), tuple(c + self.radius))

    def _level(self, x: np.ndarray) -> np.ndarray:
        x = points(x, 2)
        return np.sum((x - self.center) ** 2, axis=-1) / self.radius**2

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed disk."""
        return self._level(x) <= 1.0

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open disk."""
        return self._level(x) < 1.0

    def sample_boundary(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points on the circle, uniform in angle.

        Parameters
        ----------
        n
            Number of points, at least 0.
        rng
            The random generator.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)``.
        """
        angle = rng.uniform(0.0, 2.0 * np.pi, size=integer("n", n, 0))
        circle = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        return np.asarray(self.center) + self.radius * circle

    def boundary_distance(self, x: np.ndarray) -> np.ndarray:
        """Distance to the circle, ``radius - |x - center|``, for points in the disk."""
        x = points(x, 2)
        return self.radius - np.linalg.norm(x - np.asarray(self.center), axis=-1)


@dataclass(frozen=True)
class CutQuarterDisk:
    """The quarter of the unit disk in the first quadrant, cut by a line.

    The convex set ``x > 0, y > 0, x^2 + y^2 < 1, y > x - 1/2`` of
    Nystrom-Vestberg 2023, Example 4.6: two straight sides on the axes, a
    third on the line ``y = x - 1/2`` from ``(1/2, 0)`` to the circle at
    ``((1 + sqrt 7)/4, (sqrt 7 - 1)/4)``, and an arc of the unit circle from
    there to ``(0, 1)``. It has no parameters.
    """

    @property
    def dim(self) -> int:
        """Space dimension, always 2."""
        return 2

    @property
    def bbox(self) -> Box:
        """The box ``[0, (1 + sqrt 7)/4] x [0, 1]``."""
        return Box((0.0, 0.0), ((1.0 + np.sqrt(7.0)) / 4.0, 1.0))

    def _margins(self, x: np.ndarray) -> np.ndarray:
        """Shape ``(..., 4)``: each defining inequality as ``margin > 0``."""
        x = points(x, 2)
        a, b = x[..., 0], x[..., 1]
        return np.stack([a, b, 1.0 - a**2 - b**2, b - a + 0.5], axis=-1)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed domain."""
        return np.all(self._margins(x) >= 0.0, axis=-1)

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open domain."""
        return np.all(self._margins(x) > 0.0, axis=-1)

    def sample_boundary(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points on the four sides, in proportion to their lengths.

        The sides, in this order: the segment from ``(0, 0)`` to ``(1/2, 0)``
        (length 1/2), the cut from ``(1/2, 0)`` to the circle (length
        ``(sqrt 7 - 1) / (2 sqrt 2)``), the arc from there to ``(0, 1)``, and
        the segment from ``(0, 1)`` to ``(0, 0)`` (length 1). Points on the
        arc are uniform in angle.

        Parameters
        ----------
        n
            Number of points, at least 0.
        rng
            The random generator.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)``.
        """
        corner = np.array([(1.0 + np.sqrt(7.0)) / 4.0, (np.sqrt(7.0) - 1.0) / 4.0])
        start = np.array([0.5, 0.0])
        angle = np.arctan2(corner[1], corner[0])
        lengths = np.array(
            [0.5, np.linalg.norm(corner - start), np.pi / 2.0 - angle, 1.0]
        )
        bottom, cut, arc, left = (
            rng.uniform(0.0, 1.0, size=(count, 1))
            for count in _allocate(integer("n", n, 0), lengths)
        )
        theta = angle + (np.pi / 2.0 - angle) * arc
        return np.concatenate(
            [
                np.hstack([0.5 * bottom, np.zeros_like(bottom)]),
                start + cut * (corner - start),
                np.hstack([np.cos(theta), np.sin(theta)]),
                np.hstack([np.zeros_like(left), left]),
            ]
        )

    def boundary_distance(self, x: np.ndarray) -> np.ndarray:
        """Distance to the boundary, for points in the closed domain.

        The domain is the intersection of two half-planes, a third half-plane
        and the unit disk, so the distance from a point inside is the least
        of its distances to the four boundaries: ``x``, ``y``,
        ``(y - x + 1/2) / sqrt 2`` and ``1 - |x|``.
        """
        x = points(x, 2)
        a, b = x[..., 0], x[..., 1]
        return np.min(
            np.stack(
                [a, b, (b - a + 0.5) / np.sqrt(2.0), 1.0 - np.hypot(a, b)], axis=-1
            ),
            axis=-1,
        )
