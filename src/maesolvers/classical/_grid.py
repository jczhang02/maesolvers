"""Grid helpers shared by the finite-difference schemes of this package."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from maesolvers.core.domains import Field


def interior_mask(n: int) -> np.ndarray:
    """Shape ``(n + 1, n + 1)``: True at the nodes off the boundary."""
    mask = np.zeros((n + 1, n + 1), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


def spline(x: np.ndarray, y: np.ndarray, values: np.ndarray, degree: int) -> Field:
    """Tensor spline of the given degree through values on the grid ``x`` by ``y``."""
    fit = RectBivariateSpline(
        x, y, values.reshape(len(x), len(y)), kx=degree, ky=degree
    )

    def u(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        out = fit(p[..., 0].ravel(), p[..., 1].ravel(), grid=False)
        return out.reshape(p.shape[:-1])

    return u
