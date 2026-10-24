"""The semi-Lagrangian scheme of Feng and Jensen (2017) on a structured grid.

Feng and Jensen (SIAM J. Numer. Anal. 55 (2017), Sections 3 and 4) write the
Monge-Ampere equation ``det D^2 u = f`` as the Hamilton-Jacobi-Bellman
(HJB) equation

    sup over B >= 0, tr B = 1, of  -B : D^2 u + 2 sqrt(f) sqrt(det B) = 0,

which holds exactly for convex ``u`` and needs no convexity constraint: a
solution is convex because ``-B : D^2 u <= 0`` for every ``B``. With
``B = a s1 s1^T + (1 - a) s2 s2^T`` for an orthogonal frame ``(s1, s2)``,
``B : D^2 u`` is ``a d1 + (1 - a) d2`` with ``d1, d2`` the second
derivatives along the frame, approximated by the semi-Lagrangian
differences

    d(x) = (u(x + k s) - 2 u(x) + u(x - k s)) / k^2,

whose ends ``x +/- k s`` lie anywhere in the domain and take the value of
the piecewise linear interpolant of the nodal values there. The weights of
the interpolant are nonnegative, so the scheme is monotone, and its
solutions converge to the convex viscosity solution.

The supremum over the split ``a`` is taken in closed form
(:func:`frame_values`); the frames are those of a fixed set of directions
(:func:`stencil_directions`). The numerical kernels are module-level pure
functions; :class:`SemiLagrangian` composes them into a
:class:`~maesolvers.classical.schemes.Scheme` that
:class:`~maesolvers.classical.iterations.Howard` solves.

The paper works on unstructured triangular grids; here the nodes are those
of the finite-difference family (a uniform grid over the bounding box and
the points where its lines cross a curved boundary, as
:func:`~maesolvers.classical.schemes.shortley_weller_nodes` places them),
and the piecewise linear interpolant lives on their Delaunay triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay

from maesolvers.classical._linear import solve_m_matrix
from maesolvers.classical.schemes import (
    Scheme,
    _nearest_on_segments,
    shortley_weller_nodes,
)
from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import (
    Box,
    CutQuarterDisk,
    Disk,
    DiskUnionSquare,
    Domain,
    Ellipse,
    Field,
)
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError


def stencil_directions(count: int) -> np.ndarray:
    """Return the unit directions at the angles ``i pi / count``, ``0 <= i < count``.

    Feng and Jensen (2017, Section 8) discretise the rotations of the
    frames by the angles ``i pi / 64``. With an even ``count`` the
    perpendicular of direction ``i`` is direction ``i + count / 2``, so the
    directions form ``count / 2`` orthogonal frames: frame ``p`` is the
    pair of directions ``p`` and ``p + count / 2``. At least 4 directions
    (two frames) are needed: with the axis frame alone the scheme is
    consistent only for Hessians aligned with the axes, and
    :func:`fit_hessian` needs three pairwise independent directions.

    Parameters
    ----------
    count
        Number of directions, even and at least 4.

    Returns
    -------
    numpy.ndarray
        Shape ``(count, 2)``; direction 0 is ``(1, 0)`` and direction
        ``count / 2`` is ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``count`` is not an even integer of at least 4.
    """
    count = integer("directions", count, 4)
    if count % 2:
        raise ValueError(f"directions must be even, got {count}")
    angles = np.pi * np.arange(count) / count
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def frame_values(d1: np.ndarray, d2: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return the HJB operator restricted to one frame, in closed form.

    For the second differences ``d1``, ``d2`` along the two directions of a
    frame,

        sup over 0 <= a <= 1 of  -(a d1 + (1 - a) d2) + 2 sqrt(f) sqrt(a (1 - a))
            = sqrt(((d1 - d2) / 2)^2 + f) - (d1 + d2) / 2,

    by Cauchy-Schwarz with ``a = (1 + s) / 2``. It is at most zero exactly
    when ``d1 d2 >= f`` and ``d1 + d2 >= 0``, and it is nonincreasing in
    ``d1`` and ``d2``, so the scheme is monotone.

    Parameters
    ----------
    d1, d2
        The second differences, of one shape.
    f
        The right-hand side ``f >= 0``, broadcast against them.

    Returns
    -------
    numpy.ndarray
    """
    return np.sqrt((0.5 * (d1 - d2)) ** 2 + f) - 0.5 * (d1 + d2)


def frame_weights(d1: np.ndarray, d2: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return the split ``a`` that attains the supremum of :func:`frame_values`.

    ``a = (1 - (d1 - d2) / (2 r)) / 2`` with ``r = sqrt(((d1 - d2) / 2)^2 + f)``,
    the weight of ``d1``; ``1 - a`` weighs ``d2``. They are the derivatives
    of minus :func:`frame_values` in ``d1`` and ``d2``. Where ``r = 0``
    (``d1 = d2`` and ``f = 0``) every split attains it, and ``a = 1/2``.

    Parameters
    ----------
    d1, d2, f
        As in :func:`frame_values`.

    Returns
    -------
    numpy.ndarray
        In ``[0, 1]``.
    """
    half = 0.5 * (d1 - d2)
    r = np.sqrt(half**2 + f)
    ratio = np.divide(half, r, out=np.zeros(np.broadcast(half, r).shape), where=r > 0)
    return 0.5 * (1.0 - ratio)


def fit_hessian(directions: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Fit a symmetric matrix to second differences along directions.

    The least-squares solution of ``s_i^T H s_i = second_i`` over the
    directions ``s_i``, exact for the second differences of a quadratic.

    Parameters
    ----------
    directions
        Shape ``(D, 2)``, unit vectors, at least three of them pairwise
        independent.
    second
        Shape ``(D, m)``: the second differences at ``m`` points.

    Returns
    -------
    numpy.ndarray
        Shape ``(m, 2, 2)``.
    """
    c, s = directions[:, 0], directions[:, 1]
    design = np.stack([c * c, 2.0 * c * s, s * s], axis=-1)
    (hxx, hxy, hyy), *_ = np.linalg.lstsq(design, second, rcond=None)
    return np.stack([np.stack([hxx, hxy], -1), np.stack([hxy, hyy], -1)], -2)


def linear_weights(tri: Delaunay, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the piecewise linear interpolation weights of points in a triangulation.

    A point that lies on the boundary of the convex hull in exact arithmetic
    can come out outside it by rounding, such as a stencil end whose reach
    is exactly its distance to the hull. A point outside by at most
    ``1e-9`` times the size of the triangulation takes the weights of the
    nearest point of the hull's boundary, interpolated along that edge, so
    the weights stay nonnegative and the scheme monotone.

    Parameters
    ----------
    tri
        A Delaunay triangulation in the plane.
    points
        Shape ``(q, 2)``, in the convex hull of ``tri.points``.

    Returns
    -------
    tuple of numpy.ndarray
        The vertex indices and the barycentric weights of the triangle that
        holds each point, or of the hull edge next to it, each of shape
        ``(q, 3)``; the weights are nonnegative and add up to 1.

    Raises
    ------
    ValueError
        If a point lies outside the triangulation by more than rounding.
    """
    simplex = tri.find_simplex(points)
    outside = simplex < 0
    transform = tri.transform[simplex]
    first = np.einsum("qij,qj->qi", transform[:, :2], points - transform[:, 2])
    weights = np.concatenate([first, 1.0 - first.sum(axis=1, keepdims=True)], axis=1)
    # A point on an edge may come out a rounding error outside; clip, so the
    # weights stay nonnegative and the scheme monotone.
    weights = np.clip(weights, 0.0, None)
    vertices = tri.simplices[simplex]
    if np.any(outside):
        start, end = (
            tri.points[tri.convex_hull[:, 0]],
            tri.points[tri.convex_hull[:, 1]],
        )
        edge, s = _nearest_on_segments(points[outside], start, end)
        gap = np.linalg.norm(
            start[edge] + s[:, None] * (end[edge] - start[edge]) - points[outside],
            axis=-1,
        )
        size = np.max(np.ptp(tri.points, axis=0))
        if np.any(gap > _ROUNDING * size):
            raise ValueError(
                f"{np.count_nonzero(gap > _ROUNDING * size)} points lie outside "
                "the triangulation"
            )
        hull = tri.convex_hull[edge]
        vertices[outside] = np.stack([hull[:, 0], hull[:, 1], hull[:, 0]], axis=-1)
        weights[outside] = np.stack([1.0 - s, s, np.zeros_like(s)], axis=-1)
    return vertices, weights / weights.sum(axis=1, keepdims=True)


_ROUNDING = 1e-9
"""How far, relative to the triangulation's size, :func:`linear_weights` lets a
point lie outside it and still interpolates it from the nearest hull edge."""


def semi_lagrangian_operators(
    tri: Delaunay, unknown: np.ndarray, directions: np.ndarray, reach: np.ndarray
) -> sp.csr_matrix:
    """Return the semi-Lagrangian second differences at the unknowns.

    At the node ``x`` of index ``unknown[i]`` and the direction ``s``, the
    difference is ``(I u(x + k s) - 2 u(x) + I u(x - k s)) / k^2`` with
    ``k = reach[i]`` and ``I u`` the piecewise linear interpolant on ``tri``.

    Parameters
    ----------
    tri
        The Delaunay triangulation of all the nodes.
    unknown
        Shape ``(m,)``: the indices of the nodes that carry unknowns.
    directions
        Shape ``(D, 2)``, unit vectors.
    reach
        Shape ``(m,)``, positive: the stencil size ``k`` at each unknown,
        with ``x +/- k s`` in the triangulation.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(D m, N)``, ``N`` the number of nodes: row ``d m + i`` maps
        the values at all nodes to the difference along direction ``d`` at
        unknown ``i``.
    """
    m, total = unknown.size, len(tri.points)
    centre = tri.points[unknown]
    scale = 1.0 / reach**2
    rows = np.arange(m)
    blocks = []
    for s in directions:
        cols = [unknown]
        vals = [-2.0 * scale]
        for sign in (1.0, -1.0):
            vertices, weights = linear_weights(tri, centre + sign * reach[:, None] * s)
            cols.append(vertices.ravel())
            vals.append((weights * scale[:, None]).ravel())
        block = sp.csr_matrix(
            (
                np.concatenate(vals),
                (
                    np.concatenate([rows, np.tile(np.repeat(rows, 3), 2)]),
                    np.concatenate(cols),
                ),
            ),
            shape=(m, total),
        )
        blocks.append(block)
    return sp.csr_matrix(sp.vstack(blocks))


@dataclass(frozen=True, eq=False)
class SemiLagrangianGrid:
    """One problem discretised by :class:`SemiLagrangian`, its working context.

    Only :class:`SemiLagrangian` reads the fields; they are not a stable
    interface and may change with the scheme. The unknowns are the values at
    the ``m`` nodes that :meth:`SemiLagrangian.place_nodes` marks interior;
    the ``b`` boundary nodes carry the Dirichlet data.

    Attributes
    ----------
    h
        The grid spacing ``(hx, hy)``.
    nodes
        Shape ``(m + b, 2)``, from :meth:`SemiLagrangian.place_nodes`: by
        default the grid nodes in the open domain, then the boundary points.
    interior
        Shape ``(m + b,)``, boolean: True at the ``m`` nodes with unknowns.
    f
        The right-hand side at the unknowns.
    boundary_values
        The Dirichlet data at the boundary points.
    directions
        Shape ``(D, 2)``, from :func:`stencil_directions`.
    reach
        Shape ``(m,)``: the stencil size ``k`` at each unknown.
    operators
        Shape ``(D m, m)``: the columns of :func:`semi_lagrangian_operators`
        that act on the unknowns.
    data
        Shape ``(D, m)``: the contribution of the Dirichlet data.
    axis_pair
        Shape ``(m, m)``: ``d_0 + d_(D/2)``, the axis frame's differences on
        the unknowns.
    tri
        The Delaunay triangulation of ``nodes``.
    """

    h: tuple[float, float]
    nodes: np.ndarray
    interior: np.ndarray
    f: np.ndarray
    boundary_values: np.ndarray
    directions: np.ndarray
    reach: np.ndarray
    operators: sp.csr_matrix
    data: np.ndarray
    axis_pair: sp.csr_matrix
    tri: Delaunay


@dataclass(frozen=True, kw_only=True)
class SemiLagrangian(Scheme):
    """The semi-Lagrangian scheme of Feng and Jensen (2017) on a 2D convex domain.

    At every unknown the discrete equation is

        max over frames (s1, s2) of  sqrt(((d1 - d2) / 2)^2 + f) - (d1 + d2) / 2 = 0,

    the HJB form of ``det D^2 u = f`` (:func:`frame_values`), with
    ``d1``, ``d2`` the semi-Lagrangian second differences along the frame:
    the stencil ends ``x +/- k s`` take the piecewise linear interpolant of
    the nodal values. A root has ``d1 d2 >= f`` and ``d1, d2 >= 0`` in every
    frame, so it is convex along every stencil direction.

    The stencil size is ``k = stencil_factor * h`` away from the boundary
    and the distance to the boundary of the triangulated domain near it, so
    that the stencil stays inside (Feng-Jensen 2017, Remark 4.1 and
    Section 8). The paper's nodes are those of an unstructured
    triangulation; here they are the grid nodes in the open domain and the
    boundary points of
    :func:`~maesolvers.classical.schemes.shortley_weller_nodes` (the grid
    nodes on the boundary and the points where grid lines and diagonals
    cross it), triangulated by Delaunay; a subclass places other nodes by
    overriding :meth:`place_nodes`.

    Parameters
    ----------
    n
        Cells per side of the grid over the domain's bounding box
        (``h = 1/n`` on the unit square, ADR 0001), at least 2.
    stencil_factor
        ``m`` in ``k = m h``, positive, with ``h`` the larger spacing of the
        grid. The paper's best ``m`` grows as the grid is refined, from 2 to
        16 on its smooth example and from 4 to 64 on its nonsmooth one;
        convergence needs ``k -> 0`` and ``h / k -> 0``.
    directions
        Number of stencil directions, evenly spaced in angle over
        ``[0, pi)``, even and at least 4, so they form ``directions / 2``
        orthogonal frames (:func:`stencil_directions`); the paper's 64.

    Attributes
    ----------
    domains : tuple of type
        ``(Box, Disk, Ellipse, CutQuarterDisk, DiskUnionSquare)``, the convex
        domains: the triangulation covers the convex hull of the nodes.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2,)``.
    """

    domains: ClassVar[tuple[type, ...]] = (
        Box,
        Disk,
        Ellipse,
        CutQuarterDisk,
        DiskUnionSquare,
    )
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2,)

    n: int = 64
    stencil_factor: float = 4.0
    directions: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", integer("n", self.n, 2))
        object.__setattr__(
            self, "stencil_factor", positive("stencil_factor", self.stencil_factor)
        )
        object.__setattr__(self, "directions", len(stencil_directions(self.directions)))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem's domain, boundary and dimension are declared.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return (
            isinstance(problem.domain, self.domains)
            and isinstance(problem.boundary, self.boundaries)
            and problem.dim in self.dims
        )

    def place_nodes(self, domain: Domain) -> tuple[np.ndarray, np.ndarray]:
        """Place the nodes and mark those that carry unknowns.

        By default the nodes of
        :func:`~maesolvers.classical.schemes.shortley_weller_nodes`, but a
        grid node closer than ``1e-3 h`` (``h`` the larger spacing) to the
        boundary of the convex hull of all nodes carries the Dirichlet data
        at its position instead of an unknown (:func:`clear_of_hull`): its
        stencil, cut to that distance, would have second differences of
        order ``1 / k^2`` that rounding swamps. On a curved boundary a grid
        node can come arbitrarily close to the hull: on T6 at ``n = 100``
        one lies ``3.6e-6 h`` from it. A subclass that overrides this method
        changes the nodes and keeps the rest of the scheme. The stencil
        size still uses the spacing of the grid of ``n`` cells per side over
        ``domain.bbox``.

        Parameters
        ----------
        domain
            A domain the scheme supports.

        Returns
        -------
        tuple of numpy.ndarray
            ``nodes``, shape ``(N, 2)``, and ``interior``, shape ``(N,)``,
            True at the nodes that carry unknowns; the others carry the
            Dirichlet data. The nodes must span the domain, since the
            triangulation of all of them is where the stencil ends go, and
            the unknowns must lie strictly inside their convex hull, or
            :meth:`discretize` raises ``ValueError``.
        """
        nodes, interior = shortley_weller_nodes(domain, self.n)
        floor = _MIN_REACH * max(_spacing(domain, self.n))
        return nodes, clear_of_hull(nodes, interior, floor)

    def discretize(self, problem: Problem) -> SemiLagrangianGrid:
        """Place the nodes, triangulate them and build the second differences.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        SemiLagrangianGrid

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        ValueError
            If an override of :meth:`place_nodes` marks an unknown on the
            boundary of the convex hull of the nodes (within ``1e-9`` times
            their extent; the default nodes keep ``1e-3 h`` from it), or if
            ``f`` is negative or not finite at an unknown.
        """
        if not self.supports(problem):
            raise UnsupportedProblemError(
                f"SemiLagrangian does not support {problem.name}: it needs a 2D "
                "Box, Disk, Ellipse, CutQuarterDisk or DiskUnionSquare with a "
                "Dirichlet boundary"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        nodes, interior = self.place_nodes(problem.domain)
        h = _spacing(problem.domain, self.n)
        tri = Delaunay(nodes)
        unknown = np.flatnonzero(interior)
        inside = _hull_distance(tri.points, tri.convex_hull, nodes[unknown])
        # A node on the hull has distance 0 up to rounding; a stencil of that
        # size has no second difference. The default nodes keep _MIN_REACH h
        # from the hull, so only an override of place_nodes gets here.
        on_hull = ~(inside > _ROUNDING * np.max(np.ptp(nodes, axis=0)))
        if np.any(on_hull):
            raise ValueError(
                f"{np.count_nonzero(on_hull)} of the unknowns that "
                f"{type(self).__name__}.place_nodes marks on {problem.name} lie "
                "on the boundary of the convex hull of all nodes, where the "
                "stencil has no room; an unknown must lie strictly inside it"
            )
        reach = np.minimum(self.stencil_factor * max(h), inside)
        f = problem.f(nodes[unknown])
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError(
                f"SemiLagrangian needs a finite f >= 0 at the unknowns, and "
                f"{problem.name} has f from {np.min(f)} to {np.max(f)} there"
            )
        directions = stencil_directions(self.directions)
        full = semi_lagrangian_operators(tri, unknown, directions, reach)
        operators = full[:, interior].tocsr()
        boundary_values = problem.boundary.g(nodes[~interior])
        data = (full[:, ~interior] @ boundary_values).reshape(len(directions), -1)
        half = len(directions) // 2
        m = unknown.size
        axis_pair = operators[:m] + operators[half * m : (half + 1) * m]
        return SemiLagrangianGrid(
            h=(float(h[0]), float(h[1])),
            nodes=nodes,
            interior=interior,
            f=f,
            boundary_values=boundary_values,
            directions=directions,
            reach=reach,
            operators=operators,
            data=data,
            axis_pair=axis_pair,
            tri=tri,
        )

    def rhs(self, grid: SemiLagrangianGrid) -> np.ndarray:
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

    def second_differences(self, grid: SemiLagrangianGrid, v: np.ndarray) -> np.ndarray:
        """Return the second differences along every direction at the unknowns.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        numpy.ndarray
            Shape ``(D, m)``, in the order of :func:`stencil_directions`.
        """
        return (grid.operators @ v).reshape(grid.data.shape) + grid.data

    def residual(self, grid: SemiLagrangianGrid, v: np.ndarray) -> np.ndarray:
        """Return the max over frames of :func:`frame_values` at the unknowns.

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
        d1, d2 = _frames(self.second_differences(grid, v))
        return frame_values(d1, d2, grid.f).max(axis=0)

    def jacobian(self, grid: SemiLagrangianGrid, v: np.ndarray) -> sp.csc_matrix:
        """Return the Jacobian of :meth:`residual` through its maximising frame.

        At each unknown the frame that attains the maximum is differentiated
        (Danskin's theorem): minus its split ``a``
        (:func:`frame_weights`) times the difference along its first
        direction, minus ``1 - a`` times that along the second. That is the
        linear part ``L^B`` of the maximising control ``B``, an M-matrix
        (Feng-Jensen 2017, Lemma 5.1), and a Newton step with it is a step
        of Howard's algorithm.

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
        d1, d2 = _frames(self.second_differences(grid, v))
        frames, m = d1.shape
        best = frame_values(d1, d2, grid.f).argmax(axis=0)
        cells = np.arange(m)
        a = frame_weights(d1[best, cells], d2[best, cells], grid.f)
        rows = np.concatenate([cells, cells])
        cols = np.concatenate([best * m + cells, (best + frames) * m + cells])
        collapse = sp.csr_matrix(
            (np.concatenate([-a, a - 1.0]), (rows, cols)),
            shape=(m, 2 * frames * m),
        )
        return sp.csc_matrix(collapse @ grid.operators)

    def solve_laplace(self, grid: SemiLagrangianGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve ``d_x + d_y = rhs`` with the Dirichlet data for the unknowns.

        ``d_x`` and ``d_y`` are the scheme's second differences along the
        axes. With ``rhs = 2 sqrt(f)`` this is the equation of the control
        ``B = Id / 2``, Feng and Jensen's start for Howard's algorithm.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        rhs
            The right-hand side at the unknowns.

        Returns
        -------
        numpy.ndarray
        """
        half = len(grid.directions) // 2
        return solve_m_matrix(grid.axis_pair, rhs - grid.data[0] - grid.data[half])

    def solution(self, grid: SemiLagrangianGrid, v: np.ndarray) -> Solution:
        """Node values, the fitted Hessian, the interpolant and the final residual.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        Solution
            ``u`` is the piecewise linear interpolant on the triangulation,
            the scheme's own function, and outside the triangulation (in
            slivers along a curved boundary) the value at the nearest point
            of its boundary, the paper's constant extension along the
            normal. ``node_hessian`` at an unknown is the least-squares fit
            to its second differences along every direction
            (:func:`fit_hessian`), NaN at the boundary points. A fit to
            nonnegative second differences can be indefinite, worst where
            they are nearly all 0 (flat regions, kinks), so ``evaluate``'s
            ``nonconvex_fraction`` and ``eigmin`` measure the fit: a root is
            convex along every stencil direction, yet on T40 at ``n = 80``,
            ``m = 16``, exact to rounding, a quarter of the nodes have a
            fitted eigenvalue below ``-1e-6``
            (``benchmarks/feng_jensen2017.md``, Finding 3). ``info`` holds
            the spacing ``h``, ``residual_max``, the max absolute residual,
            ``min_reach``, the smallest stencil size over ``h``, and
            ``interpolant``, ``"linear"``.
        """
        values = np.empty(len(grid.nodes))
        values[grid.interior] = v
        values[~grid.interior] = grid.boundary_values
        hess = np.full((len(grid.nodes), 2, 2), np.nan)
        hess[grid.interior] = fit_hessian(
            grid.directions, self.second_differences(grid, v)
        )
        return Solution(
            u=_linear(grid.tri, values),
            nodes=grid.nodes,
            values=values,
            node_hessian=hess,
            interior=grid.interior,
            unknowns=int(v.size),
            info={
                "h": grid.h,
                "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
                "min_reach": float(np.min(grid.reach) / max(grid.h)),
                "interpolant": "linear",
            },
        )


def clear_of_hull(nodes: np.ndarray, interior: np.ndarray, floor: float) -> np.ndarray:
    """Unmark the unknowns closer than ``floor`` to the hull of all nodes.

    The stencil of an unknown is cut to its distance from the boundary of
    the convex hull of the nodes, so an unknown very close to it has second
    differences of order ``1 / distance^2``, which rounding swamps. Such a
    node carries the Dirichlet data at its position instead, as
    :func:`~maesolvers.classical.schemes.shortley_weller_nodes` does with a
    grid node within ``1e-6`` steps of the boundary along a grid line.

    Parameters
    ----------
    nodes
        Shape ``(N, 2)``, all the nodes.
    interior
        Shape ``(N,)``, boolean: True at the nodes that carry unknowns.
    floor
        The least distance, nonnegative, from an unknown to the boundary of
        the convex hull of ``nodes``.

    Returns
    -------
    numpy.ndarray
        ``interior``, False at the unknowns closer than ``floor``; a copy
        if any changes.

    Raises
    ------
    ValueError
        If no unknown is left.
    """
    unknown = np.flatnonzero(interior)
    close = _hull_distance(nodes, ConvexHull(nodes).simplices, nodes[unknown]) < floor
    if not np.any(close):
        return interior
    if np.all(close):
        raise ValueError(f"no unknown lies {floor} or farther from the hull")
    interior = interior.copy()
    interior[unknown[close]] = False
    return interior


_MIN_REACH = 1e-3
"""The least distance, in steps of the larger grid spacing, from an unknown of
the default nodes to the boundary of their convex hull (:func:`clear_of_hull`)."""


def _spacing(domain: Domain, n: int) -> tuple[float, float]:
    """Return the spacing ``(hx, hy)`` of ``n`` cells per side over the bbox."""
    box = domain.bbox
    return ((box.hi[0] - box.lo[0]) / n, (box.hi[1] - box.lo[1]) / n)


def _frames(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(D, m)`` second differences into the frames' two directions."""
    half = d.shape[0] // 2
    return d[:half], d[half:]


_PAIRS_PER_BLOCK = 2**22
"""Point-edge pairs per block of :func:`_hull_distance`: some 32 MB of temporaries."""


def _hull_distance(
    nodes: np.ndarray, hull: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Distance from each point inside the convex hull of ``nodes`` to its boundary.

    ``hull`` holds the hull's edges, pairs of indices into ``nodes``. Inside
    a convex polygon the nearest boundary point lies on the line of the
    nearest edge, so the distance is the least distance to the lines of the
    hull's edges. The points go through in blocks, so memory stays bounded
    however many there are.
    """
    a, b = nodes[hull[:, 0]], nodes[hull[:, 1]]
    normal = np.stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    offset = np.sum(normal * a, axis=-1)
    # Orient every normal outwards: the mean of the nodes lies inside.
    outward = np.where(normal @ nodes.mean(axis=0) < offset, 1.0, -1.0)
    normal, offset = normal * outward[:, None], offset * outward
    distance = np.empty(len(points))
    size = max(1, _PAIRS_PER_BLOCK // len(offset))
    for lo in range(0, len(points), size):
        block = points[lo : lo + size]
        distance[lo : lo + size] = np.min(offset - block @ normal.T, axis=1)
    return distance


def _linear(tri: Delaunay, values: np.ndarray) -> Field:
    """Piecewise linear interpolant on ``tri``, constant along normals outside it."""
    fit = LinearNDInterpolator(tri, values)
    hull = tri.convex_hull
    start, end = hull[:, 0], hull[:, 1]

    def u(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        flat = p.reshape(-1, 2)
        out = fit(flat)
        outside = tri.find_simplex(flat) < 0
        if np.any(outside):
            edge, s = _nearest_on_segments(
                flat[outside], tri.points[start], tri.points[end]
            )
            out[outside] = (1.0 - s) * values[start[edge]] + s * values[end[edge]]
        return out.reshape(p.shape[:-1])

    return u
