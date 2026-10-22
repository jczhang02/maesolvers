"""The standard 9-point discretisation on a uniform grid over a 2D box.

Shared by :class:`~maesolvers.solvers.FDNewton` and
:class:`~maesolvers.solvers.FDPoisson` until the scheme and the iteration
become separate parts (ADR 0004).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import factorized

from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Field
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution


@dataclass(frozen=True, eq=False)
class Grid:
    """``n`` cells per side, ``(n + 1)^2`` nodes, Dirichlet data at the boundary.

    Node ``(i, j)`` has flat index ``i * (n + 1) + j``; ``i`` runs along x.
    The difference operators are split into the columns that act on the
    interior unknowns (``a*``) and on the known boundary values (``b*``).
    """

    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    interior: np.ndarray
    boundary: np.ndarray
    f: np.ndarray
    boundary_values: np.ndarray
    axx: sp.csc_matrix
    ayy: sp.csc_matrix
    axy: sp.csc_matrix
    bxx: sp.csc_matrix
    byy: sp.csc_matrix
    bxy: sp.csc_matrix
    solve_laplace: Callable[[np.ndarray], np.ndarray]
    """Solves ``laplace u = rhs`` at the interior nodes with zero boundary values."""

    @property
    def h(self) -> tuple[float, float]:
        """Grid spacing along x and y."""
        return float(self.x[1] - self.x[0]), float(self.y[1] - self.y[0])

    def second_derivatives(
        self, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``u_xx``, ``u_yy``, ``u_xy`` at the interior nodes for unknowns ``v``."""
        ub = self.boundary_values
        return (
            self.axx @ v + self.bxx @ ub,
            self.ayy @ v + self.byy @ ub,
            self.axy @ v + self.bxy @ ub,
        )

    def solve_poisson(self, rhs: np.ndarray) -> np.ndarray:
        """Unknowns ``v`` with ``laplace u = rhs`` and the Dirichlet data."""
        lap_b = (self.bxx + self.byy) @ self.boundary_values
        return self.solve_laplace(rhs - lap_b)

    def values(self, v: np.ndarray) -> np.ndarray:
        """The solution at every node: unknowns inside, Dirichlet data outside."""
        u = np.empty(len(self.nodes))
        u[self.interior] = v
        u[self.boundary] = self.boundary_values
        return u

    def node_hessian(self, v: np.ndarray) -> np.ndarray:
        """The discrete Hessian at every node, NaN at the boundary nodes."""
        uxx, uyy, uxy = self.second_derivatives(v)
        hess = np.full((len(self.nodes), 2, 2), np.nan)
        hess[self.interior, 0, 0] = uxx
        hess[self.interior, 1, 1] = uyy
        hess[self.interior, 0, 1] = hess[self.interior, 1, 0] = uxy
        return hess

    def residual(self, v: np.ndarray) -> np.ndarray:
        """``u_xx u_yy - u_xy^2 - f`` at the interior nodes."""
        uxx, uyy, uxy = self.second_derivatives(v)
        return uxx * uyy - uxy**2 - self.f

    def interpolant(self, v: np.ndarray, degree: int) -> Field:
        """Tensor spline of the given degree through the node values."""
        m = len(self.x)
        spline = RectBivariateSpline(
            self.x, self.y, self.values(v).reshape(m, m), kx=degree, ky=degree
        )

        def u(p: np.ndarray) -> np.ndarray:
            p = np.asarray(p, dtype=float)
            out = spline(p[..., 0].ravel(), p[..., 1].ravel(), grid=False)
            return out.reshape(p.shape[:-1])

        return u


def build_grid(problem: Problem, n: int) -> Grid:
    """The grid with ``n`` cells per side over the problem's 2D box.

    Parameters
    ----------
    problem
        A problem on a 2D :class:`~maesolvers.core.domains.Box` with a
        :class:`~maesolvers.core.boundary.Dirichlet` boundary.
    n
        Cells per side, at least 2.

    Returns
    -------
    Grid
    """
    if not isinstance(problem.boundary, Dirichlet):
        raise TypeError("build_grid needs a Dirichlet problem")
    box = problem.domain.bbox
    m = n + 1
    x = np.linspace(box.lo[0], box.hi[0], m)
    y = np.linspace(box.lo[1], box.hi[1], m)
    nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    mask = np.zeros((m, m), dtype=bool)
    mask[1:-1, 1:-1] = True
    interior = np.flatnonzero(mask.ravel())
    boundary = np.flatnonzero(~mask.ravel())
    hx, hy = x[1] - x[0], y[1] - y[0]
    dxx, dyy, dxy = _difference_operators(m, hx, hy, interior)
    (axx, bxx), (ayy, byy), (axy, bxy) = (
        (d[:, interior].tocsc(), d[:, boundary].tocsc()) for d in (dxx, dyy, dxy)
    )
    return Grid(
        x=x,
        y=y,
        nodes=nodes,
        interior=interior,
        boundary=boundary,
        f=problem.f(nodes[interior]),
        boundary_values=problem.boundary.g(nodes[boundary]),
        axx=axx,
        ayy=ayy,
        axy=axy,
        bxx=bxx,
        byy=byy,
        bxy=bxy,
        solve_laplace=factorized((axx + ayy).tocsc()),
    )


def grid_parameters(
    n: object, tol: object, spline_degree: object
) -> tuple[int, float, int]:
    """Validate and normalise the parameters both grid solvers share.

    Returns
    -------
    tuple
        ``n``, ``tol`` and ``spline_degree`` as plain int, float and int.

    Raises
    ------
    TypeError
        If ``n`` or ``spline_degree`` is not an integer or ``tol`` not a real.
    ValueError
        If a value is out of range.
    """
    n = integer("n", n, 2)
    tol = positive("tol", tol)
    spline_degree = integer("spline_degree", spline_degree, 1)
    if spline_degree > min(5, n):
        raise ValueError(
            f"spline_degree must be between 1 and min(5, n) = {min(5, n)}, "
            f"got {spline_degree}"
        )
    return n, tol, spline_degree


def grid_solution(grid: Grid, v: np.ndarray, spline_degree: int) -> Solution:
    """The solution fields of unknowns ``v``: nodes, values, Hessian, spline."""
    return Solution(
        u=grid.interpolant(v, spline_degree),
        nodes=grid.nodes,
        values=grid.values(v),
        node_hessian=grid.node_hessian(v),
        unknowns=int(grid.interior.size),
        info={
            "h": grid.h,
            "residual_max": float(np.max(np.abs(grid.residual(v)))),
        },
    )


def _difference_operators(
    m: int, hx: float, hy: float, interior: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Centred ``u_xx``, ``u_yy``, ``u_xy`` from all nodes to interior nodes."""
    rows = np.arange(interior.size)

    def build(offsets: list[tuple[int, int, float]], scale: float) -> sp.csr_matrix:
        r, c, v = [], [], []
        for di, dj, w in offsets:
            r.append(rows)
            c.append(interior + di * m + dj)
            v.append(np.full(rows.size, w * scale))
        return sp.csr_matrix(
            (np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
            shape=(interior.size, m * m),
        )

    dxx = build([(1, 0, 1.0), (0, 0, -2.0), (-1, 0, 1.0)], 1.0 / hx**2)
    dyy = build([(0, 1, 1.0), (0, 0, -2.0), (0, -1, 1.0)], 1.0 / hy**2)
    dxy = build(
        [(1, 1, 1.0), (1, -1, -1.0), (-1, 1, -1.0), (-1, -1, 1.0)],
        1.0 / (4.0 * hx * hy),
    )
    return dxx, dyy, dxy
