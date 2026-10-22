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
from typing import Any, ClassVar, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CloughTocher2DInterpolator, RectBivariateSpline
from scipy.sparse.linalg import factorized

from maesolvers.core._params import integer
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box, CutQuarterDisk, Disk, Domain, Ellipse, Field
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError


class Scheme(ABC):
    """Base class of every scheme (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005); an iteration checks this on
    construction, as the solver base class checks a solver. It implements the four
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
        """Build the grid of one problem: nodes, operators and data.

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
        """Return the right-hand side ``f`` at the unknowns.

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
        """Return the solution fields of the result for the unknowns ``v``.

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
    """Apply the 2D Monge-Ampere operator ``u_xx u_yy - u_xy^2`` to second derivatives."""
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


_SW_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
"""Grid directions of the Shortley-Weller second differences, in index steps."""

_SNAP = 1e-6
"""A node closer to the boundary than this many steps joins the boundary."""

_BISECTIONS = 64
"""Halvings that locate a boundary crossing on a grid segment, past float64 precision."""


class ShortleyWellerOperators(NamedTuple):
    """What :func:`shortley_weller_operators` returns; it unpacks as a tuple.

    Attributes
    ----------
    nodes : numpy.ndarray
        Shape ``(k + m, 2)``: the ``k`` grid nodes that carry unknowns, in the
        flat order of :func:`difference_operators`, then the ``m`` boundary
        points.
    interior : numpy.ndarray
        Shape ``(k + m,)``, boolean: True at the first ``k``.
    dxx, dyy, dxy : scipy.sparse.csr_matrix
        Shape ``(k, k + m)``: from the values at ``nodes`` to the differences
        at the ``k`` nodes.
    min_step : numpy.ndarray
        Shape ``(k,)``: the smallest step fraction (``a`` or ``b``) in the
        stencil of each of the ``k`` nodes, 1 where every neighbour is in the
        open domain. The weights grow like ``1 / min_step``.
    """

    nodes: np.ndarray
    interior: np.ndarray
    dxx: sp.csr_matrix
    dyy: sp.csr_matrix
    dxy: sp.csr_matrix
    min_step: np.ndarray


def shortley_weller_operators(domain: Domain, n: int) -> ShortleyWellerOperators:
    """Second differences on a uniform grid cut by a curved boundary.

    The grid has ``n`` cells per side over the domain's bounding box, and its
    nodes in the open domain carry the unknowns. At each of them the second
    difference along a grid direction ``v`` (the two axes and the two
    diagonals, in index steps) uses the neighbour ``P +/- v`` when it lies in
    the open domain, and otherwise the point ``P +/- t v``, ``0 < t <= 1``,
    where the segment towards that neighbour leaves the domain, found by
    bisection on ``domain.interior``. Through ``u_-``, ``u_P``, ``u_+`` at
    ``-b v``, 0, ``a v`` the difference is the second derivative of the
    interpolating parabola (Shortley-Weller 1938),

        D_v u = 2 / (a b (a + b)) (b u_+ - (a + b) u_P + a u_-),

    exact for quadratics; ``u_xx = D_(1,0) / hx^2``, ``u_yy = D_(0,1) / hy^2``
    and ``u_xy = (D_(1,1) - D_(1,-1)) / (4 hx hy)``. Where ``a = b = 1`` these
    are the stencils of :func:`difference_operators`.

    The boundary points carry the Dirichlet data: the grid nodes on the
    boundary (in the closed but not in the open domain), the nodes on either
    side of the boundary closer to it than ``1e-6`` steps along a grid
    direction (rounding puts nodes that lie on a circle in exact arithmetic,
    such as ``(15/17, 8/17)`` or ``(0.6, 0.8)`` on the unit circle, on either
    side of it; a node outside would otherwise come back as a crossing from
    each of its neighbours), the nodes on the edges of the bounding box that
    the closed domain contains (the box touches the domain there, and rounding
    may put such a node in the open domain), and the crossings. A crossing is
    looked for on the segment to an outside neighbour only, so a boundary that
    crosses a grid segment twice (a non-convex domain with features finer than
    the grid) is not resolved.

    Parameters
    ----------
    domain
        A 2D domain.
    n
        Cells per side of the grid over ``domain.bbox``, at least 2.

    Returns
    -------
    ShortleyWellerOperators
        The ``k`` nodes with unknowns and the ``m`` boundary points, the
        mask of the first, the three operators from the values at all of
        them to the differences at the first, and the smallest step fraction
        of each stencil.

    Raises
    ------
    ValueError
        If no grid node carries an unknown.
    """
    box = domain.bbox
    x = np.linspace(box.lo[0], box.hi[0], n + 1)
    y = np.linspace(box.lo[1], box.hi[1], n + 1)
    h = np.array([x[1] - x[0], y[1] - y[0]])
    grid = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    inside = domain.interior(grid)
    # The edges of the bounding box only touch the closed domain, so a node
    # there joins the boundary even if it tests inside by rounding; every
    # node left with an unknown then has its eight neighbours in the grid.
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    on_boundary = domain.contains(grid) & ~inside
    # The step fraction to the boundary along each half-direction, 1 where the
    # neighbour is in the closed domain.
    halves = [(sign * di, sign * dj) for di, dj in _SW_DIRECTIONS for sign in (1, -1)]
    ii, jj = np.nonzero(inside)
    t = np.ones((len(halves), ii.size))
    for d, (di, dj) in enumerate(halves):
        cut = ~(inside | on_boundary)[ii + di, jj + dj]
        t[d, cut] = _crossing(domain, grid[ii[cut], jj[cut]], np.array([di, dj]) * h)
        # A neighbour outside by rounding only joins the boundary, so that its
        # other neighbours in the open domain do not add copies of it.
        far = cut & (t[d] > 1.0 - _SNAP)
        on_boundary[ii[far] + di, jj[far] + dj] = True
        t[d, far] = 1.0
    near = np.min(t, axis=0) < _SNAP
    inside[ii[near], jj[near]] = False
    on_boundary[ii[near], jj[near]] = True
    ii, jj, t = ii[~near], jj[~near], t[:, ~near]
    k = ii.size
    if k == 0:
        raise ValueError(f"no node of the {n} x {n} grid lies in the open domain")

    # Columns: the unknowns, then the boundary nodes, then the crossings in
    # the order they are met.
    column = np.full(inside.shape, -1)
    column[ii, jj] = np.arange(k)
    bi, bj = np.nonzero(on_boundary)
    column[bi, bj] = k + np.arange(bi.size)
    points = [grid[ii, jj], grid[bi, bj]]
    m = bi.size
    columns = np.empty((len(halves), k), dtype=int)
    for d, (di, dj) in enumerate(halves):
        columns[d] = column[ii + di, jj + dj]
        cut = columns[d] < 0
        start = grid[ii[cut], jj[cut]]
        points.append(start + t[d, cut, None] * (np.array([di, dj]) * h))
        columns[d, cut] = k + m + np.arange(start.shape[0])
        m += start.shape[0]

    def directional(index: int, scale: float) -> sp.csr_matrix:
        """``scale * D_v`` for the direction ``_SW_DIRECTIONS[index]``."""
        a, b = t[2 * index], t[2 * index + 1]
        values = np.concatenate(
            [2.0 / (a * (a + b)), -2.0 / (a * b), 2.0 / (b * (a + b))]
        )
        cols = np.concatenate(
            [columns[2 * index], np.arange(k), columns[2 * index + 1]]
        )
        rows = np.tile(np.arange(k), 3)
        return sp.csr_matrix((scale * values, (rows, cols)), shape=(k, k + m))

    cross = 1.0 / (4.0 * h[0] * h[1])
    dxx = directional(0, 1.0 / h[0] ** 2)
    dyy = directional(1, 1.0 / h[1] ** 2)
    dxy = (directional(2, cross) - directional(3, cross)).tocsr()
    dxy.eliminate_zeros()  # the centre of D_xy cancels at a full stencil
    interior = np.zeros(k + m, dtype=bool)
    interior[:k] = True
    return ShortleyWellerOperators(
        np.concatenate(points), interior, dxx, dyy, dxy, np.min(t, axis=0)
    )


def _crossing(domain: Domain, start: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Where each segment ``start + t step``, ``0 <= t <= 1``, leaves the open domain.

    Each ``start`` lies in the open domain and ``start + step`` outside the
    closed one; the returned ``t`` in ``(0, 1]`` is the upper end of the
    bisection bracket, a point not in the open domain.
    """
    lo = np.zeros(len(start))
    hi = np.ones(len(start))
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        inner = domain.interior(start + mid[:, None] * step)
        lo = np.where(inner, mid, lo)
        hi = np.where(inner, hi, mid)
    return hi


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
    h
        Grid spacing ``(hx, hy)``.
    nodes
        On a box, shape ``((n + 1)^2, 2)``, node ``(i, j)`` at flat index
        ``i * (n + 1) + j``; on a curved domain, the nodes and boundary points
        of :func:`shortley_weller_operators`.
    interior
        Boolean, one per node: the nodes that carry unknowns.
    f
        The right-hand side at the unknowns.
    row_scale
        The factor of each equation in :meth:`Standard9.residual`, one per
        unknown: the ``min_step`` of :func:`shortley_weller_operators`, 1 on
        a box.
    boundary_values
        The Dirichlet data at the boundary nodes, in their order in ``nodes``.
    axx, ayy, axy, bxx, byy, bxy
        The blocks of :func:`difference_operators` or
        :func:`shortley_weller_operators`.
    laplace
        Solves ``(axx + ayy) v = rhs`` by a sparse LU factorisation.
    interpolate
        Builds the solution off the nodes from the values at ``nodes``.
    interpolant
        What :attr:`interpolate` builds: ``"spline(k)"`` for the tensor
        spline of degree ``k`` on a box, ``"clough_tocher"`` otherwise.
    """

    h: tuple[float, float]
    nodes: np.ndarray
    interior: np.ndarray
    f: np.ndarray
    row_scale: np.ndarray
    boundary_values: np.ndarray
    axx: sp.csc_matrix
    ayy: sp.csc_matrix
    axy: sp.csc_matrix
    bxx: sp.csc_matrix
    byy: sp.csc_matrix
    bxy: sp.csc_matrix
    laplace: Callable[[np.ndarray], np.ndarray]
    interpolate: Callable[[np.ndarray], Field]
    interpolant: str


@dataclass(frozen=True, kw_only=True)
class Standard9(Scheme):
    """The standard 9-point scheme on a uniform grid over a 2D domain.

    Centred second differences, Dirichlet data imposed at the boundary nodes
    (Benamou-Froese-Oberman 2010). On a curved domain the grid covers the
    bounding box, the nodes in the open domain carry the unknowns, and next to
    the boundary the differences take unequal steps to the points where the
    grid lines and diagonals cross it, with the Dirichlet data there
    (Shortley-Weller, :func:`shortley_weller_operators`). The scheme is
    consistent but not monotone: its discrete equation has non-convex roots as
    well, and which root a solve reaches depends on the iteration
    (``benchmarks/standard_fd.md``, Finding 2).

    Parameters
    ----------
    n
        Cells per side of the grid over the domain's bounding box
        (``h = 1/n`` on the unit square, ADR 0001), at least 2; on a box
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes on a box, 1 to 5; a grid with fewer than ``spline_degree``
        cells per side takes degree ``n``. On a curved domain the solution
        off the nodes is the piecewise cubic Clough-Tocher interpolant,
        whatever the degree. ``Result.info["interpolant"]`` names the one
        used.

    Attributes
    ----------
    domains : tuple of type
        ``(Box, Disk, Ellipse, CutQuarterDisk)``.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2,)``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box, Disk, Ellipse, CutQuarterDisk)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2,)

    n: int = 64
    spline_degree: int = 3

    def __post_init__(self) -> None:
        n = integer("n", self.n, 2)
        spline_degree = integer("spline_degree", self.spline_degree, 1)
        if spline_degree > 5:
            raise ValueError(
                f"spline_degree must be between 1 and 5, got {spline_degree}"
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
            True for a 2D :class:`Box`, :class:`Disk`, :class:`Ellipse` or
            :class:`CutQuarterDisk` with a :class:`Dirichlet` boundary.
        """
        return (
            isinstance(problem.domain, self.domains)
            and isinstance(problem.boundary, self.boundaries)
            and problem.dim in self.dims
        )

    def discretize(self, problem: Problem) -> Grid:
        """Build the grid with ``n`` cells per side over the problem's bounding box.

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
                f"Standard9 does not support {problem.name}: it needs a 2D Box, "
                "Disk, Ellipse or CutQuarterDisk with a Dirichlet boundary"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        n, box = self.n, problem.domain.bbox
        x = np.linspace(box.lo[0], box.hi[0], n + 1)
        y = np.linspace(box.lo[1], box.hi[1], n + 1)
        if isinstance(problem.domain, Box):
            nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
            interior = _interior_mask(n).ravel()
            dxx, dyy, dxy = difference_operators(n, x[1] - x[0], y[1] - y[0])
            row_scale = np.ones(np.count_nonzero(interior))
            degree = min(self.spline_degree, n)  # a spline needs degree + 1 knots
            interpolant = f"spline({degree})"

            def interpolate(values: np.ndarray) -> Field:
                return _spline(x, y, values, degree)

        else:
            nodes, interior, dxx, dyy, dxy, row_scale = shortley_weller_operators(
                problem.domain, n
            )
            interpolant = "clough_tocher"

            def interpolate(values: np.ndarray) -> Field:
                return _scattered(nodes, values)

        (axx, bxx), (ayy, byy), (axy, bxy) = (
            (d[:, interior].tocsc(), d[:, ~interior].tocsc()) for d in (dxx, dyy, dxy)
        )
        return Grid(
            h=(float(x[1] - x[0]), float(y[1] - y[0])),
            nodes=nodes,
            interior=interior,
            f=problem.f(nodes[interior]),
            row_scale=row_scale,
            boundary_values=problem.boundary.g(nodes[~interior]),
            axx=axx,
            ayy=ayy,
            axy=axy,
            bxx=bxx,
            byy=byy,
            bxy=bxy,
            laplace=factorized((axx + ayy).tocsc()),
            interpolate=interpolate,
            interpolant=interpolant,
        )

    def rhs(self, grid: Grid) -> np.ndarray:
        """Return the right-hand side ``f`` at the unknowns.

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
        """``t (u_xx u_yy - u_xy^2 - f)`` at the interior nodes.

        ``t`` is 1 on a box and at every node whose stencil lies in the open
        domain. Next to a curved boundary it is the smallest step fraction of
        the node's Shortley-Weller stencil: the weights there grow like
        ``1 / t``, and so does the rounding error of the equation, which would
        otherwise keep a residual tolerance out of reach. Scaling an equation
        changes neither its roots nor a Newton direction.

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
        return grid.row_scale * (
            monge_ampere(*self.second_derivatives(grid, v)) - grid.f
        )

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
        t = grid.row_scale
        uxx, uyy, uxy = self.second_derivatives(grid, v)
        return monge_ampere_jacobian(
            t * uxx, t * uyy, t * uxy, grid.axx, grid.ayy, grid.axy
        )

    def solve_laplace(self, grid: Grid, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplace u = rhs`` with the Dirichlet data for the unknowns.

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
        """Node values, the discrete Hessian, the interpolant and the final residual.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        Solution
            ``u`` is the spline or the Clough-Tocher interpolant through the
            node values. ``info`` holds the spacing ``h``; ``residual_max``,
            the max absolute value of :meth:`residual` (row-scaled next to a
            curved boundary); ``min_step``, the smallest step fraction of a
            stencil (1 on a box); and ``interpolant``, ``"spline(k)"`` or
            ``"clough_tocher"``.
        """
        values = np.empty(len(grid.nodes))
        values[grid.interior] = v
        values[~grid.interior] = grid.boundary_values
        uxx, uyy, uxy = self.second_derivatives(grid, v)
        hess = np.full((len(grid.nodes), 2, 2), np.nan)
        hess[grid.interior, 0, 0] = uxx
        hess[grid.interior, 1, 1] = uyy
        hess[grid.interior, 0, 1] = hess[grid.interior, 1, 0] = uxy
        return Solution(
            u=grid.interpolate(values),
            nodes=grid.nodes,
            values=values,
            node_hessian=hess,
            interior=grid.interior,
            unknowns=int(np.count_nonzero(grid.interior)),
            info={
                "h": grid.h,
                "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
                "min_step": float(np.min(grid.row_scale)),
                "interpolant": grid.interpolant,
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


def _scattered(nodes: np.ndarray, values: np.ndarray) -> Field:
    """Clough-Tocher interpolant through values at scattered nodes in the plane.

    Inside the convex hull of the nodes it is scipy's piecewise cubic
    interpolant. The domain reaches past the hull: along a smooth boundary in
    slivers of width ``O(h^2)`` between consecutive boundary points, at a
    corner of the domain that is not a grid node in a wedge of width
    ``O(h)``. A point ``q`` there is extrapolated to first order from its
    projection ``p`` on the nearest hull edge, ``u(p) + grad u(p) . (q - p)``:
    ``u`` linear along the edge, and ``grad u`` linear between the gradients
    the Clough-Tocher interpolant estimates at the edge's two nodes. Both
    are ``O(h)``-accurate or better, so the error is ``O(h^2)`` in the
    wedges as in the slivers. A node that Qhull leaves out of the
    triangulation (one of two nodes some ``1e-5`` steps apart) has NaN
    gradients, but it is not a vertex of the hull, so they are never read.
    """
    spline = CloughTocher2DInterpolator(nodes, values)
    # The gradients at the nodes, shape (k, 1, 2): an attribute scipy sets
    # on construction but does not document; the corner tests pin its use.
    grad = np.asarray(spline.grad).reshape(len(nodes), 2)
    hull = spline.tri.convex_hull
    start, end = hull[:, 0], hull[:, 1]

    def u(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        flat = p.reshape(-1, 2)
        out = spline(flat)
        outside = spline.tri.find_simplex(flat) < 0
        if np.any(outside):
            q = flat[outside]
            edge, s = _nearest_on_segments(q, nodes[start], nodes[end])
            i, j, s = start[edge], end[edge], s[:, None]
            on_edge = (1.0 - s) * nodes[i] + s * nodes[j]
            gradient = (1.0 - s) * grad[i] + s * grad[j]
            out[outside] = (
                (1.0 - s[:, 0]) * values[i]
                + s[:, 0] * values[j]
                + np.sum(gradient * (q - on_edge), axis=-1)
            )
        return out.reshape(p.shape[:-1])

    return u


_PAIRS_PER_BLOCK = 2**18
"""Point-segment pairs per block of :func:`_nearest_on_segments`: some 20 MB of temporaries."""


def _nearest_on_segments(
    q: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find the segment from ``a`` to ``b`` nearest each point, and where on it.

    Parameters
    ----------
    q
        Shape ``(m, 2)``: the points.
    a, b
        Shape ``(e, 2)``: the ends of the segments.

    Returns
    -------
    tuple of numpy.ndarray
        Shape ``(m,)`` each: the index of the nearest segment, and the
        fraction ``s`` in ``[0, 1]`` of the nearest point ``a + s (b - a)`` on
        it. The points go through in blocks, so memory stays bounded however
        many there are.
    """
    d = b - a
    length2 = np.sum(d**2, axis=-1)
    index = np.empty(len(q), dtype=int)
    along = np.empty(len(q))
    size = max(1, _PAIRS_PER_BLOCK // len(a))
    for lo in range(0, len(q), size):
        block = q[lo : lo + size, None, :]
        s = np.clip(np.sum((block - a) * d, axis=-1) / length2, 0.0, 1.0)
        gap = np.sum((a + s[..., None] * d - block) ** 2, axis=-1)
        nearest = np.argmin(gap, axis=1)
        index[lo : lo + size] = nearest
        along[lo : lo + size] = s[np.arange(len(nearest)), nearest]
    return index, along
