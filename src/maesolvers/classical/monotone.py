"""The monotone wide-stencil scheme of Froese and Oberman (2011).

The scheme (Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Section 3.4)
writes the Monge-Ampere operator of a convex function as the smallest
product of second derivatives along the ``d`` directions of an orthogonal
frame, and restricts the frames to the directions of a wide stencil on the
grid:

    MA_M[u] = min over frames (nu_1, ..., nu_d) of prod_k (D_nu_k nu_k u)^+

Every term is nondecreasing in the differences ``u_j - u_i``, so the scheme
is monotone (degenerate elliptic) and converges to the viscosity solution as
the grid and the angular resolution of the stencil are refined. In 2D the
frames are pairs of perpendicular directions of the 9-, 17- or 33-point
stencil; in 3D they are the four orthogonal triples of the 19-point stencil
(Section 7).

The numerical kernels are module-level pure functions: :func:`stencil_frames`,
:func:`angular_resolution`, :func:`directional_difference`, :func:`frame_values`,
:func:`frame_jacobian_coefficients`, :func:`frame_jacobian`,
:func:`monotone_kappa`, :func:`monotone_operator` and
:func:`monotone_jacobian`. :class:`Monotone17` composes them into a
:class:`~maesolvers.classical.schemes.Scheme` that
:class:`~maesolvers.classical.iterations.Newton` solves; so does
:class:`~maesolvers.classical.second_boundary.SecondBoundary`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, cast

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.spatial import SphericalVoronoi

from maesolvers.classical.grid import BoxGrid, centred_hessian, reach
from maesolvers.classical.schemes import Scheme
from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box, Field
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError

_STENCILS: dict[int, tuple[int, int]] = {9: (2, 1), 17: (2, 2), 33: (2, 3), 19: (3, 1)}
"""Stencil size in points to its dimension and its width, the largest index
offset of a neighbour."""

_DEFAULT_STENCILS: dict[int, int] = {2: 17, 3: 19}
"""The stencil :class:`Monotone17` takes in a dimension when none is given."""


def _stencil(points: int) -> tuple[int, int]:
    """Return the dimension and the width of a stencil size; ValueError for another."""
    if points not in _STENCILS:
        raise ValueError(
            f"stencil must be 9, 17 or 33 points (2D) or 19 points (3D), got {points!r}"
        )
    return _STENCILS[points]


def stencil_dimension(points: int) -> int:
    """Return the space dimension of a wide stencil: 2 for 9, 17 and 33 points, 3 for 19.

    Parameters
    ----------
    points
        9, 17, 33 or 19.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``points`` is not 9, 17, 33 or 19.
    """
    return _stencil(points)[0]


def stencil_frames(points: int) -> np.ndarray:
    """Return the orthogonal frames of grid directions of a wide stencil.

    A frame is ``d`` mutually orthogonal grid directions, one for each
    factor of the scheme's product. In 2D a stencil of width ``w`` holds
    every direction ``(a, b)`` with coprime integer components and
    ``max(|a|, |b|) <= w``, up to sign; the 9-, 17- and 33-point stencils are
    widths 1, 2 and 3 (Oberman 2008, Table 1), and each direction ``(a, b)``
    with ``a > 0`` and ``b >= 0`` is paired with its perpendicular
    ``(-b, a)``. In 3D the frames are every orthogonal triple of directions
    with components in ``{-1, 0, 1}``: the axes, and each axis with the two
    diagonals of the plane perpendicular to it. They use the three axes and
    the six face diagonals, the 19 points of the stencil of Froese-Oberman
    (2011, Section 7); a body diagonal such as ``(1, 1, 1)`` is orthogonal to
    no two directions of width 1, so the 27-point cube adds no frame.

    Parameters
    ----------
    points
        9, 17 or 33 (2D) or 19 (3D).

    Returns
    -------
    numpy.ndarray
        Shape ``(frames, d, d)`` of int: ``frames[p, k]`` is the k-th
        direction of frame ``p`` in index offsets, each direction with its
        first nonzero component positive in 3D. The axis frame comes first.
        There are 2, 4 and 8 frames (pairs) in 2D and 4 (triples) in 3D.

    Raises
    ------
    ValueError
        If ``points`` is not 9, 17, 33 or 19.
    """
    dim, width = _stencil(points)
    if dim == 2:
        first = [
            (a, b)
            for a in range(1, width + 1)
            for b in range(width + 1)
            if np.gcd(a, b) == 1
        ]
        return np.array([[(a, b), (-b, a)] for a, b in first], dtype=int)
    # Every direction up to sign once, the axes first: (1, 0, 0), (0, 1, 0), ...
    steps = range(-width, width + 1)
    directions = sorted(
        (
            v
            for v in itertools.product(steps, repeat=dim)
            if any(v) and next(c for c in v if c) > 0 and np.gcd.reduce(v) == 1
        ),
        key=lambda v: (sum(map(abs, v)), [-abs(c) for c in v], [-c for c in v]),
    )
    frames = [
        frame
        for frame in itertools.combinations(directions, dim)
        if all(np.dot(u, v) == 0 for u, v in itertools.combinations(frame, 2))
    ]
    return np.array(frames, dtype=int)


def angular_resolution(points: int) -> float:
    """Return the angular resolution ``dtheta`` of a wide stencil.

    Oberman (2008), eq. (3.3): the largest angle between a line through the
    origin and the nearest direction of the stencil. In 2D it is half the
    largest gap between neighbouring directions: ``pi/8`` for 9 points,
    ``atan(1/2)/2`` for 17 and ``atan(1/3)/2`` for 33. In 3D it is the
    covering radius of the stencil's directions on the sphere, found at the
    vertices of their spherical Voronoi diagram: ``acos(sqrt(2/3))``, about
    35.3 degrees, for 19 points, the angle between a body diagonal and the
    nearest face diagonal. :class:`~maesolvers.classical.filtered.Filtered`
    takes twice the 2D value, the full gap, as its ``dtheta``.

    Parameters
    ----------
    points
        9, 17 or 33 (2D) or 19 (3D).

    Returns
    -------
    float
        In radians.

    Raises
    ------
    ValueError
        If ``points`` is not 9, 17, 33 or 19.
    """
    dim = stencil_dimension(points)
    directions = stencil_frames(points).reshape(-1, dim)
    if dim == 2:
        angles = np.sort(np.mod(np.arctan2(directions[:, 1], directions[:, 0]), np.pi))
        gaps = np.diff(np.append(angles, angles[0] + np.pi))
        return float(np.max(gaps) / 2.0)
    unit = np.unique(directions, axis=0) / np.linalg.norm(
        np.unique(directions, axis=0), axis=1, keepdims=True
    )
    both = np.concatenate([unit, -unit])
    vertices = SphericalVoronoi(both).vertices
    cosine = np.max(vertices @ both.T, axis=1)  # to the nearest direction
    return float(np.max(np.arccos(np.clip(cosine, -1.0, 1.0))))


def directional_difference(
    grid: BoxGrid, direction: Sequence[int], g: Field
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Second difference along a grid direction at the interior nodes.

    At an interior node ``x_i`` and a direction ``nu`` (index offsets,
    spacing ``h``), let ``t+`` and ``t-`` be the fractions of the step
    ``nu h`` that stay in the box forwards and backwards, each at most 1
    (:func:`~maesolvers.classical.grid.reach`). The difference is

        D u = 2 (t- u(x_i + t+ nu h) + t+ u(x_i - t- nu h) - (t+ + t-) u(x_i))
              / (|nu|^2 h^2 t+ t- (t+ + t-)),

    the centred difference ``(u(x + nu h) - 2 u(x) + u(x - nu h)) / (|nu|^2
    h^2)`` where both arms end at a node (``t+ = t- = 1``), and a first-order
    difference with unequal arms through a point of the boundary otherwise,
    where the Dirichlet data are known (Oberman 2008, Section 3.1). An arm of
    width 1 always ends at a node.

    Parameters
    ----------
    grid
        The grid, of any dimension ``d``, with one spacing ``h = grid.h[0]``
        on every axis (a box with equal sides).
    direction
        ``d`` integer index offsets, not all zero.
    g
        The Dirichlet data, a function of points of shape ``(..., d)``.

    Returns
    -------
    tuple
        ``(A, c, full)``. ``D u = A @ v + c`` at the ``m = (n - 1)^d``
        interior nodes in flat order (:attr:`BoxGrid.interior_index
        <maesolvers.classical.grid.BoxGrid.interior_index>`; in 2D node
        ``(i, j)`` at ``(i - 1) (n - 1) + j - 1``), where ``v`` holds the
        values at the interior nodes; ``A`` is sparse of shape ``(m, m)``
        and ``c`` is the contribution of the Dirichlet data. ``full`` is True
        where both arms end at a node.
    """
    n, h = grid.n, grid.h[0]
    index = grid.interior_index
    step = np.asarray(direction)
    forward = reach(index, n, direction)
    backward = reach(index, n, [-c for c in direction])
    length2 = float(step @ step) * h * h
    scale = 2.0 / (length2 * forward * backward * (forward + backward))
    m = len(index)
    strides = (n - 1) ** np.arange(grid.dim - 1, -1, -1)
    lo = np.array([x[0] for x in grid.axes])
    rows = np.arange(m)
    # Annotated lists: numpy's stubs would otherwise pin the element type to
    # the first array's shape, and pyright rejects the appends.
    r: list[np.ndarray] = [rows]
    c: list[np.ndarray] = [rows]
    v: list[np.ndarray] = [-(forward + backward) * scale]
    data = np.zeros(m)
    arms = ((forward, 1, backward * scale), (backward, -1, forward * scale))
    for t, sign, weight in arms:
        end = index + sign * step
        node = (t == 1.0) & np.all((end > 0) & (end < n), axis=1)
        r.append(rows[node])
        c.append((end[node] - 1) @ strides)
        v.append(weight[node])
        edge = ~node
        ends = lo + (index[edge] + sign * t[edge, None] * step) * h
        data[edge] += weight[edge] * g(ends)
    operator = sp.csr_matrix(
        (np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(m, m)
    )
    return operator, data, (forward == 1.0) & (backward == 1.0)


def frame_values(d: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Return the monotone operator's value on each frame of directions.

    ``prod_k a_k^+ + kappa sum_k min(a_k, 0)`` for the ``d`` second
    differences ``a_k`` of a frame; in 2D ``a^+ b^+ + kappa (min(a, 0) +
    min(b, 0))``. Where all are nonnegative it is Froese-Oberman's
    ``prod_k (D_nu_k nu_k u)^+``; below zero it continues with slope
    ``kappa`` instead of staying flat, so it is negative exactly when the
    frame has a negative second difference. It is nondecreasing in every
    ``a_k``, so the scheme stays monotone.

    Parameters
    ----------
    d
        Shape ``(frames, dim, m)``: the second differences of each frame at
        ``m`` nodes.
    kappa
        Shape ``(m,)``, positive: the slope below zero.

    Returns
    -------
    numpy.ndarray
        Shape ``(frames, m)``.
    """
    return np.maximum(d, 0.0).prod(axis=1) + kappa * np.minimum(d, 0.0).sum(axis=1)


def frame_jacobian_coefficients(
    d: np.ndarray, kappa: np.ndarray, floor: float
) -> np.ndarray:
    """Return the derivatives of :func:`frame_values` in its second differences.

    The derivative in ``a_k`` is ``prod_{j != k} max(a_j, floor)`` where
    ``a_k >= 0`` and ``kappa`` where ``a_k < 0``; in 2D ``max(b, floor)``
    for ``a``. It is an element of the generalised (Clarke) Jacobian, with
    the other factors kept at least ``floor``, as Froese and Oberman (2011,
    Section 5.3) do, so that minus the Newton matrix stays an M-matrix where
    a factor vanishes.

    Parameters
    ----------
    d
        Shape ``(frames, dim, m)``, as in :func:`frame_values`.
    kappa
        Shape ``(m,)``, as in :func:`frame_values`.
    floor
        Positive.

    Returns
    -------
    numpy.ndarray
        Shape ``(frames, dim, m)``: entry ``[p, k]`` multiplies the k-th
        second difference of frame ``p``.
    """
    floored = np.maximum(d, floor)
    others = np.stack(
        [np.delete(floored, k, axis=1).prod(axis=1) for k in range(d.shape[1])],
        axis=1,
    )
    return np.where(d < 0.0, kappa, others)


def frame_jacobian(operators: sp.spmatrix, weights: np.ndarray) -> sp.csc_matrix:
    """Return ``sum over frames p and directions k of diag(weights[p, k]) D_pk``.

    The Jacobian of a function of the directional second differences ``D_pk v``
    whose derivative in ``D_pk v`` is ``weights[p, k]``: with
    :func:`frame_jacobian_coefficients` masked to the active frame it is
    :meth:`Monotone17.jacobian`.

    Parameters
    ----------
    operators
        ``MonotoneGrid.operators``: the matrices ``D_pk`` stacked frame by
        frame, shape ``(frames d m, m)``.
    weights
        Shape ``(frames, d, m)``.

    Returns
    -------
    scipy.sparse.csc_matrix
        Shape ``(m, m)``.
    """
    m = weights.shape[-1]
    weights = weights.ravel()
    rows = np.tile(np.arange(m), weights.size // m)
    collapse = sp.csr_matrix(
        (weights, (rows, np.arange(weights.size))), shape=(m, weights.size)
    )
    return sp.csc_matrix(collapse @ operators)


def monotone_kappa(f: np.ndarray, dim: int = 2) -> np.ndarray:
    """Return the slope ``kappa`` of :func:`frame_values` below zero.

    ``max(f, mean f, 1) ** ((d - 1) / d)``, ``sqrt(max(f, mean f, 1))`` in
    2D: at least 1, so a negative second difference ``a`` adds at least
    ``|a|`` to the residual, and of the order of the product of the other
    ``d - 1`` second differences where ``f`` is large (:class:`Monotone17`).

    Parameters
    ----------
    f
        Shape ``(m,)``: the right-hand side at the nodes of the operator.
    dim
        The dimension ``d``, 2 or 3.

    Returns
    -------
    numpy.ndarray
        Shape ``(m,)``.
    """
    scale = np.maximum(np.maximum(f, np.mean(f)), 1.0)
    return scale ** ((dim - 1) / dim)  # numpy takes sqrt for ** 0.5


def monotone_operator(
    d: np.ndarray, available: np.ndarray, kappa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the monotone operator and the frame that attains it at each node.

    The smallest :func:`frame_values` over the frames that enter the minimum
    at a node.

    Parameters
    ----------
    d
        Shape ``(frames, dim, m)``: the second differences of each frame.
    available
        Shape ``(frames, m)``, boolean: where a frame enters the minimum; at
        least one frame at every node.
    kappa
        Shape ``(m,)``, positive: the slope of :func:`frame_values`.

    Returns
    -------
    tuple
        ``(value, active)``, both shape ``(m,)``: the operator and the index
        of the first frame that attains it.
    """
    values = np.where(available, frame_values(d, kappa), np.inf)
    return values.min(axis=0), values.argmin(axis=0)


def monotone_jacobian(
    d: np.ndarray,
    active: np.ndarray,
    kappa: np.ndarray,
    floor: float,
    operators: sp.spmatrix,
) -> sp.csc_matrix:
    """Return a generalised Jacobian of :func:`monotone_operator`.

    At each node the frame that attains the minimum is differentiated
    (Danskin's theorem) with the coefficients of
    :func:`frame_jacobian_coefficients`, through :func:`frame_jacobian`.

    Parameters
    ----------
    d
        Shape ``(frames, dim, m)``, as in :func:`monotone_operator`.
    active
        Shape ``(m,)``: the frame that attains the minimum, from
        :func:`monotone_operator`.
    kappa
        Shape ``(m,)``, as in :func:`frame_values`.
    floor
        Positive, as in :func:`frame_jacobian_coefficients`.
    operators
        Shape ``(frames dim m, k)``: the second differences as a linear map
        of ``k`` unknowns, stacked as ``d.ravel()``.

    Returns
    -------
    scipy.sparse.csc_matrix
        Shape ``(m, k)``.
    """
    chosen = np.arange(d.shape[0])[:, None] == active
    weights = frame_jacobian_coefficients(d, kappa, floor)
    return frame_jacobian(operators, weights * chosen[:, None, :])


@dataclass(frozen=True, eq=False)
class MonotoneGrid:
    """One problem discretised by :class:`Monotone17`, its working context.

    Only :class:`Monotone17` reads the fields; they are not a stable
    interface and may change with the scheme. The unknowns are the values at
    the ``m = (n - 1)^d`` interior nodes; the boundary nodes carry the
    Dirichlet data.

    Attributes
    ----------
    box_grid
        The grid of ``n`` cells per side over the box; :attr:`nodes` and
        :attr:`interior` are its own.
    f
        The right-hand side at the unknowns.
    boundary_values
        The Dirichlet data at the boundary nodes, in flat order.
    frames
        The number of orthogonal frames of the stencil.
    operators
        The directional differences of :func:`directional_difference`
        stacked frame by frame, shape ``(frames d m, m)``.
    data
        Their Dirichlet contributions, shape ``(frames, d, m)``.
    available
        Shape ``(frames, m)``: where a frame enters the minimum.
    kappa
        The slope of :func:`frame_values` below zero, shape ``(m,)``:
        ``max(f, mean f, 1)^((d - 1)/d)``, ``sqrt(max(f, mean f, 1))`` in 2D.
    floor
        The Jacobian floor, ``jacobian_floor / (2 h^2)``.
    laplace
        Solves ``(D_11 + ... + D_dd) v = rhs``, the axis frame's differences,
        by a sparse LU factorisation.
    """

    box_grid: BoxGrid
    f: np.ndarray
    boundary_values: np.ndarray
    frames: int
    operators: sp.csr_matrix
    data: np.ndarray
    available: np.ndarray
    kappa: np.ndarray
    floor: float
    laplace: Callable[[np.ndarray], np.ndarray]

    @property
    def nodes(self) -> np.ndarray:
        """Shape ``((n + 1)^d, d)``: the nodes in the flat order of :class:`BoxGrid`."""
        return self.box_grid.nodes

    @property
    def interior(self) -> np.ndarray:
        """Shape ``((n + 1)^d,)``, boolean: the nodes that carry unknowns."""
        return self.box_grid.interior


@dataclass(frozen=True, kw_only=True)
class Monotone17(Scheme):
    """The monotone wide-stencil scheme of Froese and Oberman (2011) on a 2D or 3D box.

    At every interior node the discrete equation is

        min over frames of prod_k (a_k)^+ = f,

    with ``a_k`` the second differences along the ``d`` directions of an
    orthogonal frame of the stencil (:func:`stencil_frames`): pairs in 2D
    (9, 17 or 33 points), triples in 3D (19 points, Section 7). The stencil
    sets the dimension, so a scheme with a 2D stencil supports 2D problems
    only and one with the 19-point stencil 3D problems only. The scheme is
    monotone and its solutions converge to the viscosity solution; its
    accuracy is limited by the angular resolution of the stencil as well as
    by ``h``. Below zero each frame's value continues with the slope
    ``kappa = max(f, mean f, 1)^((d - 1)/d)`` (:func:`frame_values`). That
    leaves the value of every discretely convex grid function unchanged, and
    a root has no negative second difference in any frame that enters the
    minimum; without it, where ``f = 0`` any grid function with a negative
    second difference in every frame solves the equation. As ``kappa >= 1``,
    a negative second difference ``a`` adds at least ``|a|`` to the
    residual, so ``Newton``'s stopping rule cannot accept a non-convex
    iterate that is not a root.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit box, ADR 0001), so
        ``(n + 1)^d`` nodes and ``(n - 1)^d`` unknowns; at least 2.
    stencil
        Points of the stencil: 9, 17 or 33 in 2D, 19 in 3D. None (the
        default) takes the problem's dimension: 17 points in 2D, the
        19-point scheme in 3D (:meth:`resolve`).
    boundary
        What a node does with a direction whose arm leaves the box.
        ``"narrow"`` leaves the frame out of the minimum there, so the
        stencil narrows towards the boundary (the axis frame is always
        complete). ``"interpolate"`` shortens the arm to the boundary and
        takes the Dirichlet data there (Oberman 2008, Section 3.1). The
        papers describe the second; their 2D tables match the first
        (``benchmarks/monotone17.md``, Finding 1). The arms of the 9- and
        19-point stencils never leave the box, so for them the two agree.
    jacobian_floor
        Positive; :meth:`jacobian` keeps every factor of a product at least
        ``jacobian_floor / (2 h^2)``. Froese and Oberman (2011, Section 5.3)
        take ``1e-8``; in 2D that floor makes Newton converge only linearly
        where ``f = 0`` (hundreds of steps on T20 at ``n = 181``), and the
        default ``1e-12`` does not; in 3D neither is faster at every grid
        (``benchmarks/monotone19.md``). The floor changes the steps, not the
        solution.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``, as in :class:`Standard9`.

    Attributes
    ----------
    domains : tuple of type
        ``(Box,)``; :meth:`supports` also asks for equal sides.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2, 3)``, the dimensions of the stencils; :meth:`supports` asks
        for the stencil's own (:attr:`dim`).
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2, 3)

    n: int = 64
    stencil: Literal[9, 17, 19, 33] | None = None
    boundary: Literal["narrow", "interpolate"] = "narrow"
    jacobian_floor: float = 1e-12
    spline_degree: int = 3

    def __post_init__(self) -> None:
        n = integer("n", self.n, 2)
        stencil = self.stencil
        if stencil is not None:
            stencil = integer("stencil", stencil, 1)
            _stencil(stencil)  # raises for a size other than 9, 17, 19 and 33
        if self.boundary not in ("narrow", "interpolate"):
            raise ValueError(
                f"boundary must be 'narrow' or 'interpolate', got {self.boundary!r}"
            )
        spline_degree = integer("spline_degree", self.spline_degree, 1)
        if spline_degree > 5:
            raise ValueError(
                f"spline_degree must be between 1 and 5, got {spline_degree}"
            )
        object.__setattr__(self, "n", n)
        object.__setattr__(
            self, "stencil", cast(Literal[9, 17, 19, 33] | None, stencil)
        )
        object.__setattr__(
            self, "jacobian_floor", positive("jacobian_floor", self.jacobian_floor)
        )
        object.__setattr__(self, "spline_degree", spline_degree)

    @property
    def dim(self) -> int | None:
        """The space dimension of the stencil (:func:`stencil_dimension`): 2 or 3.

        None while ``stencil`` is None: the scheme then takes the problem's
        dimension (:meth:`resolve`).
        """
        return None if self.stencil is None else stencil_dimension(self.stencil)

    @property
    def angular_resolution(self) -> float:
        """The stencil's angular resolution ``dtheta`` (:func:`angular_resolution`).

        Raises
        ------
        ValueError
            While ``stencil`` is None; :meth:`resolve` a dimension first.
        """
        if self.stencil is None:
            raise ValueError(
                "Monotone17 has no angular resolution until its stencil is set "
                "or resolved for a dimension (resolve(dim))"
            )
        return angular_resolution(self.stencil)

    def resolve(self, dim: int) -> Monotone17:
        """Return the scheme with the stencil of a dimension: the default if None.

        Parameters
        ----------
        dim
            The problem's dimension, 2 or 3.

        Returns
        -------
        Monotone17
            ``self`` if ``stencil`` is set, else a copy with 17 points for
            ``dim = 2`` and 19 for ``dim = 3``.

        Raises
        ------
        ValueError
            If ``stencil`` is None and ``dim`` is neither 2 nor 3.
        """
        if self.stencil is not None:
            return self
        if dim not in _DEFAULT_STENCILS:
            raise ValueError(f"Monotone17 has no default stencil in {dim}D")
        return replace(self, stencil=cast(Literal[17, 19], _DEFAULT_STENCILS[dim]))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is a Dirichlet box with equal sides in the stencil's dimension.

        Orthogonal index directions are orthogonal in space only on square or
        cubic cells, hence the equal sides.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        dim = _DEFAULT_STENCILS.keys() if self.dim is None else (self.dim,)
        if not (
            isinstance(problem.domain, self.domains)
            and isinstance(problem.boundary, self.boundaries)
            and problem.dim in dim
        ):
            return False
        box = problem.domain.bbox
        sides = np.subtract(box.hi, box.lo)
        return bool(np.all(np.isclose(sides[1:], sides[0], rtol=1e-12, atol=0.0)))

    def discretize(self, problem: Problem) -> MonotoneGrid:
        """Build the grid and the directional differences of the stencil.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        MonotoneGrid

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        ValueError
            If ``f`` is negative or not finite at an interior node.
        """
        if not self.supports(problem):
            who = (
                "Monotone17"
                if self.stencil is None
                else f"Monotone17 with the {self.stencil}-point stencil"
            )
            needs = "a 2D or 3D" if self.dim is None else f"a {self.dim}D"
            raise UnsupportedProblemError(
                f"{who} does not support {problem.name}: it needs {needs} Box "
                "with equal sides and a Dirichlet boundary"
            )
        scheme = self.resolve(problem.dim)
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        n, box, g = self.n, problem.domain.bbox, problem.boundary.g
        box_grid = BoxGrid(box, n)
        nodes, interior, dim = box_grid.nodes, box_grid.interior, box_grid.dim
        frames = stencil_frames(cast(int, scheme.stencil))
        parts = [
            directional_difference(box_grid, direction.tolist(), g)
            for direction in frames.reshape(-1, dim)
        ]
        operators = [part[0] for part in parts]
        shape = (len(frames), dim, (n - 1) ** dim)
        full = np.array([part[2] for part in parts]).reshape(shape)
        if self.boundary == "narrow":
            available = np.asarray(full.all(axis=1))  # every direction of a frame
        else:
            available = np.ones((len(frames), (n - 1) ** dim), dtype=bool)
        f = problem.f(nodes[interior])
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError(
                f"Monotone17 needs a finite f >= 0 at the interior nodes, and "
                f"{problem.name} has f from {np.min(f)} to {np.max(f)} there"
            )
        h = box_grid.h[0]
        floor = self.jacobian_floor / (2.0 * h * h)
        laplacian = sum(operators[1:dim], operators[0])
        return MonotoneGrid(
            box_grid=box_grid,
            f=f,
            boundary_values=g(nodes[~interior]),
            frames=len(frames),
            operators=sp.csr_matrix(sp.vstack(operators)),
            data=np.array([part[1] for part in parts]).reshape(shape),
            available=available,
            kappa=monotone_kappa(f, dim),
            floor=floor,
            laplace=factorized(laplacian.tocsc()),
        )

    def rhs(self, grid: MonotoneGrid) -> np.ndarray:
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

    def directional_differences(self, grid: MonotoneGrid, v: np.ndarray) -> np.ndarray:
        """Return the second differences of every frame at the interior nodes.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        numpy.ndarray
            Shape ``(frames, d, m)``, in the order of :func:`stencil_frames`.
        """
        return (grid.operators @ v).reshape(grid.data.shape) + grid.data

    def residual(self, grid: MonotoneGrid, v: np.ndarray) -> np.ndarray:
        """``min over frames of frame_values - f`` at the interior nodes.

        On a discretely convex grid function this is Froese-Oberman's
        ``MA_M[u] - f``.

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
        d = self.directional_differences(grid, v)
        return monotone_operator(d, grid.available, grid.kappa)[0] - grid.f

    def jacobian(self, grid: MonotoneGrid, v: np.ndarray) -> sp.csc_matrix:
        """Return a generalised Jacobian ``J`` of :meth:`residual`; ``-J`` is an M-matrix.

        At each node the frame that attains the minimum is differentiated
        (Danskin's theorem) with the coefficients of
        :func:`frame_jacobian_coefficients`.

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
        d = self.directional_differences(grid, v)
        _, active = monotone_operator(d, grid.available, grid.kappa)
        return monotone_jacobian(d, active, grid.kappa, grid.floor, grid.operators)

    def solve_laplace(self, grid: MonotoneGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplace u = rhs`` with the Dirichlet data for the unknowns.

        The Laplacian is the sum of the axis frame's differences, the 5-point
        Laplacian in 2D and the 7-point one in 3D.

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
        return grid.laplace(rhs - grid.data[0].sum(axis=0))

    def solution(self, grid: MonotoneGrid, v: np.ndarray) -> Solution:
        """Node values, the centred Hessian, the spline and the final residual.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        Solution
            ``node_hessian`` holds the centred second differences
            (:func:`~maesolvers.classical.grid.centred_hessian`) at the
            interior nodes (the scheme has no Hessian of its own; in 2D these
            are the 9-point stencil's axis differences, and ``u_xy`` is half
            the difference of its two diagonal ones) and NaN at the boundary
            nodes. ``info`` holds the spacing ``h`` along each axis and
            ``residual_max``, the max absolute residual.
        """
        values = np.empty(len(grid.nodes))
        values[grid.interior] = v
        values[~grid.interior] = grid.boundary_values
        box_grid = grid.box_grid
        n, h = box_grid.n, box_grid.h[0]
        return Solution(
            # a spline needs degree + 1 knots, so a coarse grid lowers the degree
            u=box_grid.interpolant(values, min(self.spline_degree, n)),
            nodes=grid.nodes,
            values=values,
            node_hessian=centred_hessian(box_grid, values),
            interior=grid.interior,
            unknowns=int(v.size),
            info={
                "h": (h,) * box_grid.dim,
                "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
            },
        )
