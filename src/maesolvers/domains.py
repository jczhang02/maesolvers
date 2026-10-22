"""Computational domains.

Points are arrays of shape ``(..., 2)``; predicates return shape ``(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle ``[lo[0], hi[0]] x [lo[1], hi[1]]``."""

    lo: tuple[float, float]
    hi: tuple[float, float]

    def contains(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.all((x >= self.lo) & (x <= self.hi), axis=-1)

    @property
    def bbox(self) -> Box:
        return self

    def boundary_points(self, n: int) -> np.ndarray:
        """``4 n`` points along the four sides, counter-clockwise, no repeats."""
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        (x0, y0), (x1, y1) = self.lo, self.hi
        bottom = np.stack([x0 + t * (x1 - x0), np.full(n, y0)], axis=-1)
        right = np.stack([np.full(n, x1), y0 + t * (y1 - y0)], axis=-1)
        top = np.stack([x1 - t * (x1 - x0), np.full(n, y1)], axis=-1)
        left = np.stack([np.full(n, x0), y1 - t * (y1 - y0)], axis=-1)
        return np.concatenate([bottom, right, top, left])


@dataclass(frozen=True)
class Disk:
    """Closed disk with the given center and radius."""

    center: tuple[float, float]
    radius: float

    def contains(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sum((x - self.center) ** 2, axis=-1) <= self.radius**2

    @property
    def bbox(self) -> Box:
        c, r = np.asarray(self.center), self.radius
        return Box(tuple(c - r), tuple(c + r))

    def boundary_points(self, n: int) -> np.ndarray:
        """``n`` equally spaced points on the circle."""
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.asarray(self.center) + self.radius * np.stack(
            [np.cos(a), np.sin(a)], axis=-1
        )


@dataclass(frozen=True)
class Ellipse:
    """Axis-aligned ellipse, the target set of a transport boundary condition."""

    center: tuple[float, float]
    semi_axes: tuple[float, float]

    def contains(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sum(((x - self.center) / self.semi_axes) ** 2, axis=-1) <= 1.0

    def boundary_points(self, n: int) -> np.ndarray:
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.asarray(self.center) + np.asarray(self.semi_axes) * np.stack(
            [np.cos(a), np.sin(a)], axis=-1
        )
