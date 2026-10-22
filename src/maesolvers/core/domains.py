"""Computational domains and transport targets.

Points are arrays of shape ``(..., d)`` and predicates return boolean arrays of
shape ``(...)``. Every domain is a frozen dataclass that satisfies the
:class:`Domain` protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

Field = Callable[[np.ndarray], np.ndarray]
"""Maps points of shape ``(..., d)`` to values of shape ``(...)``."""

HessianField = Callable[[np.ndarray], np.ndarray]
"""Maps points of shape ``(..., d)`` to matrices of shape ``(..., d, d)``."""


@runtime_checkable
class Domain(Protocol):
    """What metrics and solvers need from a domain."""

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


def _points(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0 or x.shape[-1] != dim:
        raise ValueError(f"points must have shape (..., {dim}), got {x.shape}")
    return x


def _floats(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence of numbers") from None
    if not all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


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
        lo, hi = _floats(self.lo, "Box.lo"), _floats(self.hi, "Box.hi")
        if len(lo) == 0 or len(lo) != len(hi):
            raise ValueError(
                f"Box.lo and Box.hi need the same length >= 1, got {lo} and {hi}"
            )
        if not all(a < b for a, b in zip(lo, hi)):
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
        x = _points(x, self.dim)
        return np.all((x >= self.lo) & (x <= self.hi), axis=-1)

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open box."""
        x = _points(x, self.dim)
        return np.all((x > self.lo) & (x < self.hi), axis=-1)

    def boundary_points(self, n: int) -> np.ndarray:
        """``4 n`` points along the four sides of a 2D box, counter-clockwise.

        Parameters
        ----------
        n
            Points per side, starting at a corner, without repeats.

        Returns
        -------
        numpy.ndarray
            Shape ``(4 n, 2)``.
        """
        if self.dim != 2:
            raise ValueError(f"boundary_points needs a 2D box, this one is {self.dim}D")
        if n < 1:
            raise ValueError(f"boundary_points needs n >= 1, got {n}")
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        (x0, y0), (x1, y1) = self.lo, self.hi
        bottom = np.stack([x0 + t * (x1 - x0), np.full(n, y0)], axis=-1)
        right = np.stack([np.full(n, x1), y0 + t * (y1 - y0)], axis=-1)
        top = np.stack([x1 - t * (x1 - x0), np.full(n, y1)], axis=-1)
        left = np.stack([np.full(n, x0), y1 - t * (y1 - y0)], axis=-1)
        return np.concatenate([bottom, right, top, left])


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
        center = _floats(self.center, "Ellipse.center")
        axes = _floats(self.semi_axes, "Ellipse.semi_axes")
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
        x = _points(x, 2)
        return np.sum(((x - self.center) / np.asarray(self.semi_axes)) ** 2, axis=-1)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed ellipse."""
        return self._level(x) <= 1.0

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open ellipse."""
        return self._level(x) < 1.0

    def boundary_points(self, n: int) -> np.ndarray:
        """``n`` points on the boundary at equally spaced parameter angles.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)``, counter-clockwise from ``center + (a1, 0)``.
        """
        if n < 1:
            raise ValueError(f"boundary_points needs n >= 1, got {n}")
        angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        circle = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        return np.asarray(self.center) + np.asarray(self.semi_axes) * circle


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
        center = _floats(self.center, "Disk.center")
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
        x = _points(x, 2)
        return np.sum((x - self.center) ** 2, axis=-1) / self.radius**2

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed disk."""
        return self._level(x) <= 1.0

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open disk."""
        return self._level(x) < 1.0

    def boundary_points(self, n: int) -> np.ndarray:
        """``n`` equally spaced points on the circle.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)``, counter-clockwise from ``center + (radius, 0)``.
        """
        if n < 1:
            raise ValueError(f"boundary_points needs n >= 1, got {n}")
        angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        circle = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        return np.asarray(self.center) + self.radius * circle
