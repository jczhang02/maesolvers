"""Finite-difference schemes: the discretisation half of a classical solver.

A scheme (ADR 0004) turns a problem into discrete equations on a grid and
evaluates them; an iteration (:mod:`maesolvers.classical.iterations`) solves
them. A scheme subclasses :class:`Scheme` and is a frozen dataclass whose
fields are its hyperparameters. :meth:`Scheme.discretize` builds the grid of
one problem, and every other method takes that grid and the unknowns and
changes neither.

The difference operators and the Monge-Ampere operator are module-level pure
functions, testable without a scheme.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import factorized

from maesolvers.core._params import integer
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box, Field
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError


class Scheme(ABC):
    """Base class of every scheme (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements the four
    methods below; an iteration names the further methods it calls in
    :attr:`~maesolvers.classical.iterations.SchemeIteration.requires`, for
    example ``residual`` and ``jacobian`` for :class:`Newton
    <maesolvers.classical.iterations.Newton>`. The grid is whatever
    :meth:`discretize` returns; iterations pass it back without looking
    inside.
    """

    @abstractmethod
    def supports(self, problem: Problem) -> bool:
        """Whether the scheme discretises the problem's domain and boundary.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """

    @abstractmethod
    def discretize(self, problem: Problem) -> Any:
        """The grid of one problem: nodes, operators and data.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        Any
            Passed to every other method of the scheme.
        """

    @abstractmethod
    def rhs(self, grid: Any) -> np.ndarray:
        """The right-hand side ``f`` at the unknowns.

        Parameters
        ----------
        grid
            From :meth:`discretize`.

        Returns
        -------
        numpy.ndarray
        """

    @abstractmethod
    def solution(self, grid: Any, v: np.ndarray) -> Solution:
        """The solution fields of the result for the unknowns ``v``.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        Solution
        """


def difference_operators(
    n: int, hx: float, hy: float
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Centred second differences on a grid of ``n`` by ``n`` cells.

    Node ``(i, j)`` has flat index ``i * (n + 1) + j``, ``i`` along x. The
    stencils are the 3-point ``u_xx`` and ``u_yy`` and the 4-point ``u_xy``,
    nine points together.

    Parameters
    ----------
    n
        Cells per side, at least 2.
    hx, hy
        Spacing along x and y.

    Returns
    -------
    tuple of scipy.sparse.csr_matrix
        ``D_xx``, ``D_yy``, ``D_xy``, each of shape ``((n - 1)^2, (n + 1)^2)``:
        from the values at all nodes to the differences at the interior nodes,
        in flat order.
    """
    m = n + 1
    interior = _interior_mask(n).ravel()
    index = np.flatnonzero(interior)
    rows = np.arange(index.size)

    def build(offsets: list[tuple[int, int, float]], scale: float) -> sp.csr_matrix:
        r, c, v = [], [], []
        for di, dj, w in offsets:
            r.append(rows)
            c.append(index + di * m + dj)
            v.append(np.full(rows.size, w * scale))
        return sp.csr_matrix(
            (np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
            shape=(index.size, m * m),
        )

    dxx = build([(1, 0, 1.0), (0, 0, -2.0), (-1, 0, 1.0)], 1.0 / hx**2)
    dyy = build([(0, 1, 1.0), (0, 0, -2.0), (0, -1, 1.0)], 1.0 / hy**2)
    dxy = build(
        [(1, 1, 1.0), (1, -1, -1.0), (-1, 1, -1.0), (-1, -1, 1.0)],
        1.0 / (4.0 * hx * hy),
    )
    return dxx, dyy, dxy


def monge_ampere(uxx: np.ndarray, uyy: np.ndarray, uxy: np.ndarray) -> np.ndarray:
    """The 2D Monge-Ampere operator ``u_xx u_yy - u_xy^2`` from second derivatives."""
    return uxx * uyy - uxy**2


def monge_ampere_jacobian(
    uxx: np.ndarray,
    uyy: np.ndarray,
    uxy: np.ndarray,
    dxx: sp.spmatrix,
    dyy: sp.spmatrix,
    dxy: sp.spmatrix,
) -> sp.csc_matrix:
    """Jacobian of :func:`monge_ampere` composed with linear difference operators.

    Parameters
    ----------
    uxx, uyy, uxy
        Second differences at the current unknowns.
    dxx, dyy, dxy
        The difference operators restricted to the unknowns.

    Returns
    -------
    scipy.sparse.csc_matrix
        ``diag(u_yy) D_xx + diag(u_xx) D_yy - 2 diag(u_xy) D_xy``.
    """
    return (
        sp.diags(uyy) @ dxx + sp.diags(uxx) @ dyy - 2.0 * sp.diags(uxy) @ dxy
    ).tocsc()


def _interior_mask(n: int) -> np.ndarray:
    """Shape ``(n + 1, n + 1)``: True at the nodes off the boundary."""
    mask = np.zeros((n + 1, n + 1), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


@dataclass(frozen=True, eq=False)
class Grid:
    """One problem discretised by :class:`Standard9`, its working context.

    Only :class:`Standard9` reads the fields; they are not a stable interface
    and may change with the scheme. The unknowns are the values at the
    interior nodes; the boundary nodes carry the Dirichlet data. Every
    difference operator is split into the columns acting on the unknowns
    (``a*``) and on the boundary values (``b*``).

    Attributes
    ----------
    x, y
        Node coordinates along each axis, ``n + 1`` each.
    nodes
        Shape ``((n + 1)^2, 2)``, node ``(i, j)`` at flat index
        ``i * (n + 1) + j``.
    interior
        Shape ``((n + 1)^2,)``, boolean: the nodes that carry unknowns.
    f
        The right-hand side at the unknowns.
    boundary_values
        The Dirichlet data at the boundary nodes, in flat order.
    axx, ayy, axy, bxx, byy, bxy
        The blocks of :func:`difference_operators`.
    laplace
        Solves ``(axx + ayy) v = rhs`` by a sparse LU factorisation.
    """

    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    interior: np.ndarray
    f: np.ndarray
    boundary_values: np.ndarray
    axx: sp.csc_matrix
    ayy: sp.csc_matrix
    axy: sp.csc_matrix
    bxx: sp.csc_matrix
    byy: sp.csc_matrix
    bxy: sp.csc_matrix
    laplace: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, kw_only=True)
class Standard9(Scheme):
    """The standard 9-point scheme on a uniform grid over a 2D box.

    Centred second differences, Dirichlet data imposed at the boundary nodes
    (Benamou-Froese-Oberman 2010). The scheme is consistent and second-order
    but not monotone: its discrete equation has non-convex roots as well, and
    which root a solve reaches depends on the iteration
    (``benchmarks/standard_fd.md``, Finding 2).

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5 and at most ``n``.

    Attributes
    ----------
    domains : tuple of type
        ``(Box,)``.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2,)``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2,)

    n: int = 64
    spline_degree: int = 3

    def __post_init__(self) -> None:
        n = integer("n", self.n, 2)
        spline_degree = integer("spline_degree", self.spline_degree, 1)
        if spline_degree > min(5, n):
            raise ValueError(
                f"spline_degree must be between 1 and min(5, n) = {min(5, n)}, "
                f"got {spline_degree}"
            )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "spline_degree", spline_degree)

    def supports(self, problem: Problem) -> bool:
        """Whether the problem's domain, boundary and dimension are declared.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
            True for a 2D :class:`Box` with a :class:`Dirichlet` boundary.
        """
        return (
            isinstance(problem.domain, self.domains)
            and isinstance(problem.boundary, self.boundaries)
            and problem.dim in self.dims
        )

    def discretize(self, problem: Problem) -> Grid:
        """The grid with ``n`` cells per side over the problem's box.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        Grid

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        """
        if not self.supports(problem):
            raise UnsupportedProblemError(
                f"Standard9 does not support {problem.name}: it needs a 2D Box "
                "with a Dirichlet boundary"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        n, box = self.n, problem.domain.bbox
        x = np.linspace(box.lo[0], box.hi[0], n + 1)
        y = np.linspace(box.lo[1], box.hi[1], n + 1)
        nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
        interior = _interior_mask(n).ravel()
        dxx, dyy, dxy = difference_operators(n, x[1] - x[0], y[1] - y[0])
        (axx, bxx), (ayy, byy), (axy, bxy) = (
            (d[:, interior].tocsc(), d[:, ~interior].tocsc()) for d in (dxx, dyy, dxy)
        )
        return Grid(
            x=x,
            y=y,
            nodes=nodes,
            interior=interior,
            f=problem.f(nodes[interior]),
            boundary_values=problem.boundary.g(nodes[~interior]),
            axx=axx,
            ayy=ayy,
            axy=axy,
            bxx=bxx,
            byy=byy,
            bxy=bxy,
            laplace=factorized((axx + ayy).tocsc()),
        )

    def rhs(self, grid: Grid) -> np.ndarray:
        """The right-hand side ``f`` at the unknowns.

        Parameters
        ----------
        grid
            From :meth:`discretize`.

        Returns
        -------
        numpy.ndarray
        """
        return grid.f

    def second_derivatives(
        self, grid: Grid, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``u_xx``, ``u_yy``, ``u_xy`` at the interior nodes.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        tuple of numpy.ndarray
        """
        ub = grid.boundary_values
        return (
            grid.axx @ v + grid.bxx @ ub,
            grid.ayy @ v + grid.byy @ ub,
            grid.axy @ v + grid.bxy @ ub,
        )

    def residual(self, grid: Grid, v: np.ndarray) -> np.ndarray:
        """``u_xx u_yy - u_xy^2 - f`` at the interior nodes.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        numpy.ndarray
        """
        return monge_ampere(*self.second_derivatives(grid, v)) - grid.f

    def jacobian(self, grid: Grid, v: np.ndarray) -> sp.csc_matrix:
        """Jacobian of :meth:`residual` with respect to the unknowns.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        uxx, uyy, uxy = self.second_derivatives(grid, v)
        return monge_ampere_jacobian(uxx, uyy, uxy, grid.axx, grid.ayy, grid.axy)

    def solve_laplace(self, grid: Grid, rhs: np.ndarray) -> np.ndarray:
        """The unknowns of ``laplace u = rhs`` with the Dirichlet data.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        rhs
            The right-hand side at the interior nodes.

        Returns
        -------
        numpy.ndarray
        """
        lap_b = (grid.bxx + grid.byy) @ grid.boundary_values
        return grid.laplace(rhs - lap_b)

    def solution(self, grid: Grid, v: np.ndarray) -> Solution:
        """Node values, the discrete Hessian, the spline and the final residual.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        Solution
            ``info`` holds the spacing ``h`` and ``residual_max``, the max
            absolute residual.
        """
        values = np.empty(len(grid.nodes))
        values[grid.interior] = v
        values[~grid.interior] = grid.boundary_values
        uxx, uyy, uxy = self.second_derivatives(grid, v)
        hess = np.full((len(grid.nodes), 2, 2), np.nan)
        hess[grid.interior, 0, 0] = uxx
        hess[grid.interior, 1, 1] = uyy
        hess[grid.interior, 0, 1] = hess[grid.interior, 1, 0] = uxy
        h = (float(grid.x[1] - grid.x[0]), float(grid.y[1] - grid.y[0]))
        return Solution(
            u=_spline(grid.x, grid.y, values, self.spline_degree),
            nodes=grid.nodes,
            values=values,
            node_hessian=hess,
            interior=grid.interior,
            unknowns=int(np.count_nonzero(grid.interior)),
            info={
                "h": h,
                "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
            },
        )


def _spline(x: np.ndarray, y: np.ndarray, values: np.ndarray, degree: int) -> Field:
    """Tensor spline of the given degree through values on the grid ``x`` by ``y``."""
    spline = RectBivariateSpline(
        x, y, values.reshape(len(x), len(y)), kx=degree, ky=degree
    )

    def u(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        out = spline(p[..., 0].ravel(), p[..., 1].ravel(), grid=False)
        return out.reshape(p.shape[:-1])

    return u
