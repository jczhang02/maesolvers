"""The second boundary value problem on a square grid (Benamou-Froese-Oberman 2014).

A :class:`~maesolvers.core.boundary.Transport` problem asks that ``grad u``
map the domain ``X`` onto a convex target ``Y``. Benamou, Froese and Oberman
(J. Comput. Phys. 260 (2014), Sections 2 to 5) replace this condition by the
Hamilton-Jacobi equation ``H(grad u) = 0`` on the boundary, with ``H`` the
signed distance to the boundary of ``Y``, written through the support
function ``H*`` of ``Y`` (their Lemma 2.1) as

    H(grad u(x)) = sup over |n| = 1, n . n_x > 0 of (grad u(x) . n - H*(n)),

with ``n_x`` the outward normal of the domain at ``x``. On a grid over a
square they take the supremum over finitely many directions ``n`` and
discretise ``grad u . n`` by upwind differences that only reach into the
square (their Section 2.4), so the boundary equation is monotone. Inside the
square they solve the Monge-Ampere equation with a monotone or a filtered
scheme, ``MA[u] = f + u(x_0)``: the value of ``u`` at one node ``x_0`` added
to the right-hand side picks one of the solutions, which differ by a constant
(their Section 3). A domain that is not the square itself is embedded in it
with ``f`` extended by zero (their Section 2.3).

Every unknown is a node value, the ``(n + 1)^2`` nodes of the square grid in
the flat order of :class:`~maesolvers.classical.grid.BoxGrid`; the nodes of
the boundary ring carry the boundary equation, the interior nodes the
Monge-Ampere equation. The numerical kernels are module-level pure functions:
:func:`uniform_directions`, :func:`face_normals`, :func:`admissible`,
:func:`upwind_gradient` and :func:`directional_second_differences`.
:class:`SecondBoundary` composes them with the kernels of the monotone and
filtered schemes into a :class:`~maesolvers.classical.schemes.Scheme` that
:class:`~maesolvers.classical.iterations.Newton` solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, cast

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from maesolvers.classical.filtered import (
    filter_slope,
    filter_width,
    filtered_jacobian,
    filtered_value,
)
from maesolvers.classical.grid import (
    BoxGrid,
    centred_hessian,
    reach,
    second_differences,
)
from maesolvers.classical.monotone import (
    angular_resolution,
    monotone_jacobian,
    monotone_kappa,
    monotone_operator,
    stencil_dimension,
    stencil_frames,
)
from maesolvers.classical.schemes import Scheme, monge_ampere, monge_ampere_jacobian
from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Transport
from maesolvers.core.domains import Box, Domain
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError
from maesolvers.core.transport import knows_support_function, support_function

_ROUND = 1e-12
"""Components of a direction below this in size are set to zero."""


def uniform_directions(count: int) -> np.ndarray:
    """Return ``count`` unit directions evenly spaced in angle.

    ``n_j = (cos(2 pi j / count), sin(2 pi j / count))``, ``j = 1, ...,
    count``, the directions of Benamou-Froese-Oberman 2014, Section 2.2, with
    components below ``1e-12`` in size set to zero, so that the axis
    directions are exact when ``count`` is a multiple of 4.

    Parameters
    ----------
    count
        At least 4, so that every side and every corner of a square has an
        admissible direction (:func:`admissible`).

    Returns
    -------
    numpy.ndarray
        Shape ``(count, 2)``.
    """
    count = integer("count", count, 4)
    angle = 2.0 * np.pi * np.arange(1, count + 1) / count
    n = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    n[np.abs(n) < _ROUND] = 0.0
    return n


def face_normals() -> np.ndarray:
    """Return the outward normals of the faces of a 2D box.

    The directions in which the boundary condition of a box target is exact:
    the box is the intersection of its four supporting half-planes, so its
    signed distance and the supremum over these four directions vanish on
    the same set (Benamou-Froese-Oberman 2014, Section 6.1: "there is no
    need to discretise its boundary").

    Returns
    -------
    numpy.ndarray
        Shape ``(4, 2)``: ``(-1, 0)``, ``(1, 0)``, ``(0, -1)``, ``(0, 1)``.
    """
    return np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


def admissible(grid: BoxGrid, directions: np.ndarray) -> np.ndarray:
    """Return which directions enter the boundary equation at each ring node.

    On a side with outward normal ``n_x`` the directions with
    ``n . n_x > 0`` (Benamou-Froese-Oberman 2014, eq. (4)); at a corner,
    where the normal is not defined, the closed quadrant between the two
    faces' normals, ``n . n_x >= 0`` for both (their Section 2.4: "formal
    limits of the obliqueness constraint"). Either way the upwind
    differences of :func:`upwind_gradient` stay in the grid.

    Parameters
    ----------
    grid
        A 2D grid with square cells.
    directions
        Shape ``(k, 2)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(ring, k)``, boolean, one row per node of the boundary ring
        in flat order (``numpy.flatnonzero(~grid.interior)``).

    Raises
    ------
    ValueError
        If the grid is not 2D with square cells.
    """
    _spacing(grid)
    directions = np.asarray(directions, dtype=float)
    index = np.argwhere(~grid.interior.reshape(grid.shape))
    lower, upper = index == 0, index == grid.n
    out = np.ones((len(index), len(directions)), dtype=bool)
    corner = np.sum(lower | upper, axis=1) == 2
    for axis in range(2):
        # n . n_x for the face at the lower or upper end of this axis
        sign = np.where(lower[:, axis], -1.0, np.where(upper[:, axis], 1.0, 0.0))
        on_face = sign != 0.0
        dot = sign[:, None] * directions[None, :, axis]
        strict = np.where(corner[:, None], dot >= 0.0, dot > 0.0)
        out &= np.where(on_face[:, None], strict, True)
    return out


def upwind_gradient(
    grid: BoxGrid, nodes: np.ndarray, directions: np.ndarray
) -> sp.csr_matrix:
    """Return the upwind approximation of ``grad u . n`` at nodes, as a matrix.

    Row ``r`` approximates ``grad u . directions[r]`` at the node with flat
    index ``nodes[r]`` by

        max(n_1, 0) D_1^- u + min(n_1, 0) D_1^+ u
        + max(n_2, 0) D_2^- u + min(n_2, 0) D_2^+ u,

    with ``D^-`` and ``D^+`` the backward and forward differences
    (Benamou-Froese-Oberman 2014, Section 4.3.1): nondecreasing in the value
    at the node and nonincreasing in its neighbours, so monotone. A
    component of ``n`` that is zero takes no difference.

    Parameters
    ----------
    grid
        A 2D grid with square cells.
    nodes
        Shape ``(k,)``, flat node indices.
    directions
        Shape ``(k, 2)``, one direction per row.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(k, (n + 1)^2)``.

    Raises
    ------
    ValueError
        If the grid is not 2D with square cells, or a difference would
        leave the grid.
    """
    h = _spacing(grid)
    nodes = np.asarray(nodes)
    directions = np.asarray(directions, dtype=float)
    index = np.asarray(np.unravel_index(nodes, grid.shape)).T
    rows = np.arange(len(nodes))
    r: list[np.ndarray] = [rows]
    c: list[np.ndarray] = [nodes]
    v: list[np.ndarray] = [np.sum(np.abs(directions), axis=1) / h]
    for axis, step in enumerate((grid.offset((1, 0)), grid.offset((0, 1)))):
        component = directions[:, axis]
        backward, forward = -np.maximum(component, 0.0), np.minimum(component, 0.0)
        for sign, weight in ((-1, backward), (1, forward)):
            used = weight != 0.0
            reached = index[used, axis] + sign
            if np.any((reached < 0) | (reached > grid.n)):
                raise ValueError("an upwind difference leaves the grid")
            r.append(rows[used])
            c.append(nodes[used] + sign * step)
            v.append(weight[used] / h)
    return sp.csr_matrix(
        (np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
        shape=(len(nodes), len(grid.nodes)),
    )


def directional_second_differences(
    grid: BoxGrid, direction: tuple[int, int]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Second difference along a grid direction at the interior nodes, on all nodes.

    ``(u(x + nu h) - 2 u(x) + u(x - nu h)) / (|nu|^2 h^2)`` for the index
    offsets ``nu``, where both ends are nodes of the grid; the row of a node
    whose arm leaves the grid is zero, and the mask says where.

    Parameters
    ----------
    grid
        A 2D grid with square cells.
    direction
        Index offsets ``(a, b)``, not both zero.

    Returns
    -------
    tuple
        ``(D, full)``: ``D`` of shape ``((n - 1)^2, (n + 1)^2)``, from the
        values at all nodes to the differences at the interior nodes, and
        ``full``, shape ``((n - 1)^2,)``, True where both arms end at a node.

    Raises
    ------
    ValueError
        If the grid is not 2D with square cells.
    """
    h = _spacing(grid)
    index = grid.interior_index
    full = (reach(index, grid.n, direction) == 1.0) & (
        reach(index, grid.n, tuple(-k for k in direction)) == 1.0
    )
    centre = np.flatnonzero(grid.interior)
    rows = np.arange(len(centre))
    step = grid.offset(direction)
    scale = 1.0 / (float(np.dot(direction, direction)) * h**2)
    r = np.concatenate([rows, rows[full], rows[full]])
    c = np.concatenate([centre, centre[full] + step, centre[full] - step])
    v = np.concatenate(
        [np.where(full, -2.0 * scale, 0.0), np.full(2 * int(np.sum(full)), scale)]
    )
    shape = (len(centre), len(grid.nodes))
    return sp.csr_matrix((v, (r, c)), shape=shape), full


@dataclass(frozen=True, eq=False)
class SecondBoundaryGrid:
    """One problem discretised by :class:`SecondBoundary`, its working context.

    Only :class:`SecondBoundary` reads the fields; they are not a stable
    interface and may change with the scheme.

    Attributes
    ----------
    box_grid
        The grid over the square.
    inside
        Shape ``((n + 1)^2,)``: the nodes in the closed source domain.
    f
        ``f`` at the interior nodes, zero outside the source domain.
    anchor
        Flat index of the node ``x_0`` whose value enters every interior
        equation.
    dxx, dyy, dxy
        The centred second differences, interior rows on all nodes.
    directional
        The second differences of every pair of the monotone stencil,
        stacked, shape ``(2 pairs m, (n + 1)^2)``.
    available
        Shape ``(pairs, m)``: where a pair enters the monotone minimum.
    kappa
        The slope of :func:`~maesolvers.classical.monotone.frame_values`
        below zero, :func:`~maesolvers.classical.monotone.monotone_kappa` of
        ``f``, shape ``(m,)``.
    floor
        The monotone Jacobian floor, ``jacobian_floor / (2 h^2)``.
    width
        The filter width ``eps``, or None for the monotone interior.
    upwind
        Shape ``(k, (n + 1)^2)``: the upwind ``grad u . n`` of every
        admissible pair of a ring node and a direction, grouped by node.
    support
        Shape ``(k,)``: ``H*(n)`` of each pair.
    owner
        Shape ``(k,)``: the ring row of each pair, nondecreasing.
    neumann_data
        The Neumann data ``y_0(x) . n_x`` of :meth:`SecondBoundary.solve_laplace`,
        one per ring node.
    neumann_operator
        Shape ``(ring, (n + 1)^2)``: the upwind ``grad u . n_x`` of
        :meth:`SecondBoundary.solve_laplace`.
    """

    box_grid: BoxGrid
    inside: np.ndarray
    f: np.ndarray
    anchor: int
    dxx: sp.csr_matrix
    dyy: sp.csr_matrix
    dxy: sp.csr_matrix
    directional: sp.csr_matrix
    available: np.ndarray
    kappa: np.ndarray
    floor: float
    width: float | None
    upwind: sp.csr_matrix
    support: np.ndarray
    owner: np.ndarray
    neumann_data: np.ndarray
    neumann_operator: sp.csr_matrix

    @property
    def interior(self) -> np.ndarray:
        """Shape ``((n + 1)^2,)``: the nodes that carry the Monge-Ampere equation."""
        return self.box_grid.interior


@dataclass(frozen=True, kw_only=True)
class SecondBoundary(Scheme):
    """The second boundary value problem on a square grid (Benamou-Froese-Oberman 2014).

    Every node of the grid of ``n`` cells per side over a square carries an
    unknown. At an interior node the equation is ``MA[u] = f + u(x_0)``, with
    ``MA`` the monotone wide-stencil operator of
    :class:`~maesolvers.classical.monotone.Monotone17` (``interior="monotone"``)
    or the filtered combination of it with the standard 9-point operator
    (``interior="filtered"``, Benamou-Froese-Oberman 2014, Section 3.3), and
    ``x_0`` the node nearest the centre of the square. At a node of the
    boundary ring it is the transport condition

        max over admissible n of (upwind grad u . n - H*(n)) = 0,

    with ``H*`` the target's :func:`~maesolvers.core.transport.support_function`
    and the directions and upwind differences of :func:`admissible` and
    :func:`upwind_gradient`. A source domain that is not the square is
    embedded in it, with ``f`` zero at the nodes outside the closed domain
    (their Section 2.3). A subclass that overrides :meth:`interior_operator`
    and :meth:`interior_jacobian` puts another scheme inside the square.

    Parameters
    ----------
    n
        Cells per side of the square (``h = side / n``, ADR 0001), so
        ``(n + 1)^2`` unknowns; at least 2.
    interior
        ``"filtered"`` (the paper's computations) or ``"monotone"``.
    stencil
        Points of the monotone stencil: 9 (the paper's compact scheme), 17
        or 33. Pairs whose arms leave the square are left out of the
        minimum, as ``Monotone17(boundary="narrow")`` does.
    epsilon
        The filter width, positive; None takes the paper's
        ``sqrt(h) + dtheta`` (Section 5.2.1), with ``dtheta`` the largest
        angle between neighbouring stencil directions (``pi/4`` for 9
        points). Not used by the monotone interior.
    directions
        Number of uniform directions ``n`` in the boundary equation
        (:func:`uniform_directions`, their ``N_Y``), at least 4; None takes
        the face normals for a :class:`~maesolvers.core.domains.Box` target
        (:func:`face_normals`, exact) and 512 directions for any other.
    box
        The square the grid covers, which contains the source domain; None
        takes the domain's bounding box, which must then be a square.
    jacobian_floor
        Positive: the monotone Jacobian keeps every factor of a product at
        least ``jacobian_floor / (2 h^2)``, as in
        :class:`~maesolvers.classical.monotone.Monotone17`.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer cells per side takes degree ``n``.

    Attributes
    ----------
    domains : tuple of type
        ``(Domain,)``: any 2D domain, embedded in the square.
    boundaries : tuple of type
        ``(Transport,)``.
    dims : tuple of int
        ``(2,)``.
    """

    domains: ClassVar[tuple[type, ...]] = (Domain,)
    boundaries: ClassVar[tuple[type, ...]] = (Transport,)
    dims: ClassVar[tuple[int, ...]] = (2,)

    n: int = 64
    interior: Literal["filtered", "monotone"] = "filtered"
    stencil: Literal[9, 17, 33] = 9
    epsilon: float | None = None
    directions: int | None = None
    box: Box | None = None
    jacobian_floor: float = 1e-12
    spline_degree: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", integer("n", self.n, 2))
        if self.interior not in ("filtered", "monotone"):
            raise ValueError(
                f"interior must be 'filtered' or 'monotone', got {self.interior!r}"
            )
        stencil = integer("stencil", self.stencil, 1)
        if stencil_dimension(stencil) != 2:  # raises for an unknown size
            raise ValueError(f"stencil must be 9, 17 or 33 points, got {stencil}")
        object.__setattr__(self, "stencil", cast(Literal[9, 17, 33], stencil))
        if self.epsilon is not None:
            object.__setattr__(self, "epsilon", positive("epsilon", self.epsilon))
        if self.directions is not None:
            object.__setattr__(
                self, "directions", integer("directions", self.directions, 4)
            )
        if self.box is not None:
            if not isinstance(self.box, Box) or self.box.dim != 2:
                raise TypeError(f"box must be a 2D Box or None, got {self.box!r}")
            if not _square(self.box):
                raise ValueError(f"box must be a square, got {self.box}")
        object.__setattr__(
            self, "jacobian_floor", positive("jacobian_floor", self.jacobian_floor)
        )
        spline_degree = integer("spline_degree", self.spline_degree, 1)
        if spline_degree > 5:
            raise ValueError(
                f"spline_degree must be between 1 and 5, got {spline_degree}"
            )
        object.__setattr__(self, "spline_degree", spline_degree)

    def square(self, problem: Problem) -> Box:
        """Return the square the grid covers for a problem.

        Parameters
        ----------
        problem
            Any 2D problem.

        Returns
        -------
        Box
            :attr:`box`, or the domain's bounding box when it is None.
        """
        return self.box if self.box is not None else problem.domain.bbox

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is a 2D transport problem this scheme can grid.

        The square (:meth:`square`) must have equal sides and contain the
        domain's bounding box, and the target must be a Box, a Disk, an
        Ellipse or have the method of
        :class:`~maesolvers.core.domains.SupportFunction`.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        if not (
            isinstance(problem.boundary, Transport)
            and problem.dim in self.dims
            and isinstance(problem.domain, Domain)
        ):
            return False
        square, bbox = self.square(problem), problem.domain.bbox
        inside = np.all(
            np.asarray(square.lo) <= np.asarray(bbox.lo) + 1e-12
        ) and np.all(np.asarray(bbox.hi) <= np.asarray(square.hi) + 1e-12)
        target = problem.boundary.target
        return bool(_square(square) and inside and knows_support_function(target))

    def discretize(self, problem: Problem) -> SecondBoundaryGrid:
        """Build the grid, the interior operators and the boundary equation.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        SecondBoundaryGrid

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        ValueError
            If ``f`` is negative or not finite at an interior node of the
            domain.
        """
        if not self.supports(problem):
            raise UnsupportedProblemError(
                f"SecondBoundary does not support {problem.name}: it needs a 2D "
                "Transport problem whose domain lies in a square (the domain's "
                "bounding box or box) and a target it knows the support function of"
            )
        assert isinstance(problem.boundary, Transport)  # supports guarantees it
        target = problem.boundary.target
        box_grid = BoxGrid(self.square(problem), self.n)
        nodes, interior = box_grid.nodes, box_grid.interior
        inside = problem.domain.contains(nodes)
        f = np.zeros(int(np.sum(interior)))
        mask = inside[interior]
        f[mask] = problem.f(nodes[interior][mask])
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError(
                f"SecondBoundary needs a finite f >= 0 in the domain, and "
                f"{problem.name} has f from {np.min(f)} to {np.max(f)} there"
            )
        h = box_grid.h[0]
        d = second_differences(box_grid.n, box_grid.h)
        parts = [
            directional_second_differences(box_grid, (int(a), int(b)))
            for a, b in stencil_frames(self.stencil).reshape(-1, 2)
        ]
        pairs = len(parts) // 2
        full = np.array([part[1] for part in parts]).reshape(pairs, 2, -1)
        if self.interior == "filtered":
            if self.epsilon is None:
                dtheta = 2.0 * angular_resolution(self.stencil)
                width = filter_width(h, dtheta, divisor=1.0)
            else:
                width = self.epsilon
        else:
            width = None
        directions = self._directions(target)
        allowed = admissible(box_grid, directions)
        ring = np.flatnonzero(~interior)
        owner, which = np.nonzero(allowed)
        normal = _ring_normals(box_grid)
        centre = np.asarray(box_grid.box.lo) + 0.5 * box_grid.n * np.asarray(box_grid.h)
        tlo, thi = np.asarray(target.bbox.lo), np.asarray(target.bbox.hi)
        stretch = (thi - tlo) / (
            np.asarray(box_grid.box.hi) - np.asarray(box_grid.box.lo)
        )
        start_map = 0.5 * (tlo + thi) + (nodes[ring] - centre) * stretch
        return SecondBoundaryGrid(
            box_grid=box_grid,
            inside=inside,
            f=f,
            anchor=int(np.ravel_multi_index((box_grid.n // 2,) * 2, box_grid.shape)),
            dxx=d[0, 0],
            dyy=d[1, 1],
            dxy=d[0, 1],
            directional=sp.csr_matrix(sp.vstack([part[0] for part in parts])),
            available=full[:, 0] & full[:, 1],
            kappa=monotone_kappa(f),
            floor=self.jacobian_floor / (2.0 * h * h),
            width=width,
            upwind=upwind_gradient(box_grid, ring[owner], directions[which]),
            support=np.asarray(support_function(target, directions))[which],
            owner=owner,
            neumann_data=np.sum(start_map * normal, axis=1),
            neumann_operator=upwind_gradient(box_grid, ring, normal),
        )

    def _directions(self, target: Domain) -> np.ndarray:
        """Return the directions of the boundary equation for a target."""
        if self.directions is None:
            return (
                face_normals() if isinstance(target, Box) else uniform_directions(512)
            )
        return uniform_directions(self.directions)

    def rhs(self, grid: SecondBoundaryGrid) -> np.ndarray:
        """Return ``f`` at the interior nodes and zero on the boundary ring.

        Parameters
        ----------
        grid
            From :meth:`discretize`.

        Returns
        -------
        numpy.ndarray
            Shape ``((n + 1)^2,)``.
        """
        out = np.zeros(len(grid.interior))
        out[grid.interior] = grid.f
        return out

    def interior_operator(self, grid: SecondBoundaryGrid, v: np.ndarray) -> np.ndarray:
        """Return the Monge-Ampere operator ``MA[v]`` at the interior nodes.

        The monotone operator of
        :func:`~maesolvers.classical.monotone.monotone_operator`, or, for the
        filtered interior, its
        :func:`~maesolvers.classical.filtered.filtered_value` with the
        standard 9-point ``u_xx u_yy - u_xy^2``. With
        :meth:`interior_jacobian`, the hook a subclass overrides to put
        another scheme inside the square; the right-hand side, the anchor
        ``u(x_0)`` and the boundary ring stay as they are.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns, the values at all nodes in flat order.

        Returns
        -------
        numpy.ndarray
            Shape ``((n - 1)^2,)``, in the flat order of the interior nodes.
        """
        mono, _, _ = self._monotone(grid, v)
        if grid.width is None:
            return mono
        return filtered_value(self._accurate(grid, v), mono, grid.width)

    def interior_jacobian(
        self, grid: SecondBoundaryGrid, v: np.ndarray
    ) -> sp.csr_matrix:
        """Return a generalised Jacobian of :meth:`interior_operator`.

        The monotone rows of
        :func:`~maesolvers.classical.monotone.monotone_jacobian` (Danskin's
        theorem on the pair that attains the minimum) or, for the filtered
        interior, Froese and Oberman's
        :func:`~maesolvers.classical.filtered.filtered_jacobian` of them and
        the standard scheme's.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns, the values at all nodes in flat order.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``((n - 1)^2, (n + 1)^2)``.
        """
        mono, d, active = self._monotone(grid, v)
        rows = monotone_jacobian(d, active, grid.kappa, grid.floor, grid.directional)
        if grid.width is None:
            return sp.csr_matrix(rows)
        uxx, uyy, uxy = grid.dxx @ v, grid.dyy @ v, grid.dxy @ v
        slope = filter_slope((monge_ampere(uxx, uyy, uxy) - mono) / grid.width)
        accurate = monge_ampere_jacobian(uxx, uyy, uxy, grid.dxx, grid.dyy, grid.dxy)
        return filtered_jacobian(slope, rows, accurate)

    def _monotone(
        self, grid: SecondBoundaryGrid, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the monotone operator, its differences and active pair, inside."""
        d = (grid.directional @ v).reshape(len(grid.available), 2, -1)
        value, active = monotone_operator(d, grid.available, grid.kappa)
        return value, d, active

    def _accurate(self, grid: SecondBoundaryGrid, v: np.ndarray) -> np.ndarray:
        """Return the standard 9-point ``u_xx u_yy - u_xy^2`` at the interior nodes."""
        return monge_ampere(grid.dxx @ v, grid.dyy @ v, grid.dxy @ v)

    def _boundary(
        self, grid: SecondBoundaryGrid, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the boundary equation at each ring node and its maximising pair."""
        values = grid.upwind @ v - grid.support
        starts = np.flatnonzero(np.diff(grid.owner, prepend=-1))
        best = np.maximum.reduceat(values, starts)
        hits = np.flatnonzero(values == best[grid.owner])
        _, first = np.unique(grid.owner[hits], return_index=True)
        return best, hits[first]

    def residual(self, grid: SecondBoundaryGrid, v: np.ndarray) -> np.ndarray:
        """Return the residual of every node's equation, in flat node order.

        ``MA[v] - f - u(x_0)`` at the interior nodes, with ``MA``
        :meth:`interior_operator`; the boundary equation on the ring.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns, the values at all nodes in flat order.

        Returns
        -------
        numpy.ndarray
            Shape ``((n + 1)^2,)``.
        """
        out = np.empty(len(v))
        out[grid.interior] = self.interior_operator(grid, v) - grid.f - v[grid.anchor]
        out[~grid.interior] = self._boundary(grid, v)[0]
        return out

    def jacobian(self, grid: SecondBoundaryGrid, v: np.ndarray) -> sp.csc_matrix:
        """Return a generalised Jacobian of :meth:`residual`, in flat node order.

        The interior rows are :meth:`interior_jacobian`, with ``-1`` in the
        column of ``x_0``. A ring row is the upwind
        difference of the direction that attains the maximum
        (Benamou-Froese-Oberman 2014, Section 4.3.1).

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns, the values at all nodes in flat order.

        Returns
        -------
        scipy.sparse.csc_matrix
            Shape ``((n + 1)^2, (n + 1)^2)``.
        """
        rows = self.interior_jacobian(grid, v)
        m = len(grid.f)
        anchor = sp.csr_matrix(
            (np.full(m, -1.0), (np.arange(m), np.full(m, grid.anchor))),
            shape=(m, len(v)),
        )
        ring = grid.upwind[self._boundary(grid, v)[1]]
        return sp.csc_matrix(_in_node_order(grid, rows + anchor, ring))

    def solve_laplace(self, grid: SecondBoundaryGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve the Neumann problem of the start, ``laplace u = k rhs``.

        Benamou-Froese-Oberman 2014, Section 5.3: the 5-point Laplacian at
        the interior nodes, the Neumann data ``grad u . n_x = y_0(x) . n_x``
        of the affine map ``y_0`` of the square onto the target's bounding
        box on the ring (by the upwind differences of the outward normal,
        the diagonal at a corner), the constant ``k`` that makes the
        discrete problem solvable, and ``u(x_0) = 0``.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        rhs
            Shape ``((n + 1)^2,)``; the entries on the ring are not used.

        Returns
        -------
        numpy.ndarray
            The values at all nodes.
        """
        size = len(grid.interior)
        laplace = (grid.dxx + grid.dyy).tocsr()
        source = np.asarray(rhs, dtype=float)[grid.interior]
        rows = _in_node_order(grid, laplace, grid.neumann_operator)
        column = np.zeros(size)
        column[grid.interior] = -source
        pin = sp.csr_matrix(([1.0], ([0], [grid.anchor])), shape=(1, size))
        system = sp.bmat([[rows, sp.csr_matrix(column[:, None])], [pin, None]])
        data = np.zeros(size + 1)
        data[np.flatnonzero(~grid.interior)] = grid.neumann_data
        return np.asarray(spsolve(sp.csc_matrix(system), data))[:size]

    def solution(self, grid: SecondBoundaryGrid, v: np.ndarray) -> Solution:
        """Return the values in the domain, the centred Hessian and the spline.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns, the values at all nodes in flat order.

        Returns
        -------
        Solution
            ``nodes``, ``values``, ``node_hessian`` and ``interior`` at the
            grid nodes in the closed source domain only, where the problem
            is posed; ``interior`` marks those off the ring, where the
            centred Hessian is defined. ``u`` is the spline through every
            node of the square. ``info`` holds the spacing ``h``,
            ``residual_max``, ``u_anchor`` (``u(x_0)``, zero for exact
            mass balance) and, for the filtered interior, ``filter_width``
            and ``monotone_fraction``, the share of interior nodes where
            ``|MA_A - MA_M| > eps``.
        """
        box_grid, inside = grid.box_grid, grid.inside
        h = box_grid.h[0]
        info: dict[str, float | tuple[float, float]] = {
            "h": (h, h),
            "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
            "u_anchor": float(v[grid.anchor]),
        }
        if grid.width is not None:
            x = (self._accurate(grid, v) - self._monotone(grid, v)[0]) / grid.width
            info["filter_width"] = grid.width
            info["monotone_fraction"] = float(np.mean(np.abs(x) > 1.0))
        return Solution(
            u=box_grid.interpolant(v, min(self.spline_degree, box_grid.n)),
            nodes=box_grid.nodes[inside],
            values=v[inside],
            node_hessian=centred_hessian(box_grid, v)[inside],
            interior=box_grid.interior[inside],
            unknowns=int(v.size),
            info=info,
        )


def _in_node_order(
    grid: SecondBoundaryGrid, interior: sp.spmatrix, ring: sp.spmatrix
) -> sp.csr_matrix:
    """Return the rows of the interior and of the ring nodes, in flat node order."""
    order = np.concatenate(
        [np.flatnonzero(grid.interior), np.flatnonzero(~grid.interior)]
    )
    stacked = sp.csr_matrix(sp.vstack([interior, ring]))
    return sp.csr_matrix(stacked[np.argsort(order)])


def _spacing(grid: BoxGrid) -> float:
    """Return the spacing of a 2D grid with square cells; ValueError otherwise."""
    if grid.dim != 2 or not np.isclose(grid.h[0], grid.h[1], rtol=1e-12, atol=0.0):
        raise ValueError(f"needs a 2D grid with square cells, got spacing {grid.h}")
    return grid.h[0]


def _square(box: Box) -> bool:
    width, height = np.subtract(box.hi, box.lo)
    return bool(np.isclose(width, height, rtol=1e-12, atol=0.0))


def _ring_normals(grid: BoxGrid) -> np.ndarray:
    """Return the outward normal at each ring node, the diagonal at a corner."""
    index = np.argwhere(~grid.interior.reshape(grid.shape))
    normal = np.where(index == 0, -1.0, np.where(index == grid.n, 1.0, 0.0))
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)
