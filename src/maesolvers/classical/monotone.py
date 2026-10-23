"""The monotone wide-stencil scheme of Froese and Oberman (2011).

The scheme (Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Section 3.4)
writes the Monge-Ampere operator of a convex function as the smallest
product of second derivatives along a pair of orthogonal directions, and
restricts the pairs to the directions of a wide stencil on the grid:

    MA_M[u] = min over pairs (nu, nu') of (D_nu nu u)^+ (D_nu' nu' u)^+

Every term is nondecreasing in the differences ``u_j - u_i``, so the scheme
is monotone (degenerate elliptic) and converges to the viscosity solution as
the grid and the angular resolution of the stencil are refined.

The numerical kernels are module-level pure functions: :func:`stencil_pairs`,
:func:`angular_resolution`, :func:`directional_difference`, :func:`pair_values` and
:func:`pair_jacobian_coefficients`. :class:`Monotone17` composes them into a
:class:`~maesolvers.classical.schemes.Scheme` that
:class:`~maesolvers.classical.iterations.Newton` solves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, cast

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from maesolvers.classical.grid import BoxGrid, centred_hessian, reach
from maesolvers.classical.schemes import Scheme
from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box, Field
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError

_STENCIL_WIDTHS: dict[int, int] = {9: 1, 17: 2, 33: 3}
"""Stencil size in points to its width, the largest index offset of a neighbour."""


def stencil_pairs(points: int) -> np.ndarray:
    """Return the orthogonal pairs of grid directions of a wide stencil.

    A stencil of width ``w`` holds every direction ``(a, b)`` with coprime
    integer components and ``max(|a|, |b|) <= w``, up to sign; the 9-, 17-
    and 33-point stencils are widths 1, 2 and 3 (Oberman 2008, Table 1).
    Each direction ``(a, b)`` with ``a > 0`` and ``b >= 0`` is paired with
    its perpendicular ``(-b, a)``, which is in the stencil too.

    Parameters
    ----------
    points
        9, 17 or 33.

    Returns
    -------
    numpy.ndarray
        Shape ``(pairs, 2, 2)`` of int: ``pairs[p, k]`` is the k-th direction
        of pair ``p`` in index offsets. The axis pair ``((1, 0), (0, 1))``
        comes first. There are 2, 4 and 8 pairs.

    Raises
    ------
    ValueError
        If ``points`` is not 9, 17 or 33.
    """
    if points not in _STENCIL_WIDTHS:
        raise ValueError(f"stencil must be 9, 17 or 33 points, got {points!r}")
    width = _STENCIL_WIDTHS[points]
    first = [
        (a, b)
        for a in range(1, width + 1)
        for b in range(width + 1)
        if np.gcd(a, b) == 1
    ]
    return np.array([[(a, b), (-b, a)] for a, b in first], dtype=int)


def angular_resolution(points: int) -> float:
    """Return the angular resolution ``dtheta`` of a wide stencil.

    Oberman (2008), eq. (3.3): the largest angle between a line through the
    origin and the nearest direction of the stencil, half the largest gap
    between neighbouring directions. ``pi/8`` for 9 points, ``atan(1/2)/2``
    for 17 and ``atan(1/3)/2`` for 33.
    :class:`~maesolvers.classical.filtered.Filtered` takes twice this, the
    full gap, as its ``dtheta``.

    Parameters
    ----------
    points
        9, 17 or 33.

    Returns
    -------
    float
        In radians.

    Raises
    ------
    ValueError
        If ``points`` is not 9, 17 or 33.
    """
    directions = stencil_pairs(points).reshape(-1, 2)
    angles = np.sort(np.mod(np.arctan2(directions[:, 1], directions[:, 0]), np.pi))
    gaps = np.diff(np.append(angles, angles[0] + np.pi))
    return float(np.max(gaps) / 2.0)


def directional_difference(
    x: np.ndarray, y: np.ndarray, direction: tuple[int, int], g: Field
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Second difference along a grid direction at the interior nodes.

    At an interior node ``x_i`` and a direction ``nu`` (index offsets
    ``(a, b)``, spacing ``h``), let ``t+`` and ``t-`` be the fractions of the
    step ``nu h`` that stay in the box forwards and backwards, each at most
    1. The difference is

        D u = 2 (t- u(x_i + t+ nu h) + t+ u(x_i - t- nu h) - (t+ + t-) u(x_i))
              / (|nu|^2 h^2 t+ t- (t+ + t-)),

    the centred difference ``(u(x + nu h) - 2 u(x) + u(x - nu h)) / (|nu|^2
    h^2)`` where both arms end at a node (``t+ = t- = 1``), and a first-order
    difference with unequal arms through a point of the boundary otherwise,
    where the Dirichlet data are known (Oberman 2008, Section 3.1).

    Parameters
    ----------
    x, y
        Node coordinates along each axis, ``n + 1`` each, uniform with one
        spacing ``h`` on both axes.
    direction
        The index offsets ``(a, b)``, not both zero.
    g
        The Dirichlet data, a function of points of shape ``(..., 2)``.

    Returns
    -------
    tuple
        ``(A, c, full)``. ``D u = A @ v + c`` at the ``(n - 1)^2`` interior
        nodes in flat order (node ``(i, j)`` at ``(i - 1) (n - 1) + j - 1``),
        where ``v`` holds the values at the interior nodes; ``A`` is sparse of
        shape ``((n - 1)^2, (n - 1)^2)`` and ``c`` is the contribution of the
        Dirichlet data. ``full`` is True where both arms end at a node.
    """
    n = len(x) - 1
    h = float(x[1] - x[0])
    a, b = direction
    grid_i, grid_j = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
    i, j = grid_i.ravel(), grid_j.ravel()

    index = np.stack([i, j], axis=-1)
    forward = reach(index, n, (a, b))
    backward = reach(index, n, (-a, -b))
    length2 = float(a * a + b * b) * h * h
    scale = 2.0 / (length2 * forward * backward * (forward + backward))
    m = (n - 1) ** 2
    rows = np.arange(m)
    # Annotated lists: numpy's stubs would otherwise pin the element type to
    # the first array's shape, and pyright rejects the appends.
    r: list[np.ndarray] = [rows]
    c: list[np.ndarray] = [rows]
    v: list[np.ndarray] = [-(forward + backward) * scale]
    data = np.zeros(m)
    arms = ((forward, 1, backward * scale), (backward, -1, forward * scale))
    for t, sign, weight in arms:
        ii, jj = i + sign * a, j + sign * b
        node = (t == 1.0) & (ii > 0) & (ii < n) & (jj > 0) & (jj < n)
        r.append(rows[node])
        c.append((ii[node] - 1) * (n - 1) + jj[node] - 1)
        v.append(weight[node])
        edge = ~node
        ends = np.stack(
            [
                x[0] + (i[edge] + sign * t[edge] * a) * h,
                y[0] + (j[edge] + sign * t[edge] * b) * h,
            ],
            axis=-1,
        )
        data[edge] += weight[edge] * g(ends)
    operator = sp.csr_matrix(
        (np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(m, m)
    )
    return operator, data, (forward == 1.0) & (backward == 1.0)


def pair_values(d: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Return the monotone operator's value on each pair of directions.

    ``a^+ b^+ + kappa (min(a, 0) + min(b, 0))`` for the two second
    differences ``a`` and ``b`` of a pair. Where both are nonnegative it is
    Froese-Oberman's ``(D_nu nu u)^+ (D_nu' nu' u)^+``; below zero it
    continues with slope ``kappa`` instead of staying flat, so it is
    negative exactly when the pair has a negative second difference. It is
    nondecreasing in ``a`` and ``b``, so the scheme stays monotone.

    Parameters
    ----------
    d
        Shape ``(pairs, 2, m)``: the second differences of each pair at ``m``
        nodes.
    kappa
        Shape ``(m,)``, positive: the slope below zero.

    Returns
    -------
    numpy.ndarray
        Shape ``(pairs, m)``.
    """
    a, b = d[:, 0], d[:, 1]
    return np.maximum(a, 0.0) * np.maximum(b, 0.0) + kappa * (
        np.minimum(a, 0.0) + np.minimum(b, 0.0)
    )


def pair_jacobian_coefficients(
    d: np.ndarray, kappa: np.ndarray, floor: float
) -> np.ndarray:
    """Return the derivatives of :func:`pair_values` in its two second differences.

    The derivative in ``a`` is ``max(b, floor)`` where ``a >= 0`` and
    ``kappa`` where ``a < 0``, and likewise in ``b``: an element of the
    generalised (Clarke) Jacobian, with the other factor kept at least
    ``floor``, as Froese and Oberman (2011, Section 5.3) do, so that minus
    the Newton matrix stays an M-matrix where that factor vanishes.

    Parameters
    ----------
    d
        Shape ``(pairs, 2, m)``, as in :func:`pair_values`.
    kappa
        Shape ``(m,)``, as in :func:`pair_values`.
    floor
        Positive.

    Returns
    -------
    numpy.ndarray
        Shape ``(pairs, 2, m)``: entry ``[p, k]`` multiplies the k-th second
        difference of pair ``p``.
    """
    return np.where(d < 0.0, kappa, np.maximum(d[:, ::-1], floor))


@dataclass(frozen=True, eq=False)
class MonotoneGrid:
    """One problem discretised by :class:`Monotone17`, its working context.

    Only :class:`Monotone17` reads the fields; they are not a stable
    interface and may change with the scheme. The unknowns are the values at
    the ``m = (n - 1)^2`` interior nodes; the boundary nodes carry the
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
    pairs
        The number of orthogonal pairs of the stencil.
    operators
        The directional differences of :func:`directional_difference`
        stacked pair by pair, shape ``(2 pairs m, m)``.
    data
        Their Dirichlet contributions, shape ``(pairs, 2, m)``.
    available
        Shape ``(pairs, m)``: where a pair enters the minimum.
    kappa
        The slope of :func:`pair_values` below zero, shape ``(m,)``:
        ``sqrt(max(f, mean f, 1))``.
    floor
        The Jacobian floor, ``jacobian_floor / (2 h^2)``.
    laplace
        Solves ``(D_xx + D_yy) v = rhs`` by a sparse LU factorisation.
    """

    box_grid: BoxGrid
    f: np.ndarray
    boundary_values: np.ndarray
    pairs: int
    operators: sp.csr_matrix
    data: np.ndarray
    available: np.ndarray
    kappa: np.ndarray
    floor: float
    laplace: Callable[[np.ndarray], np.ndarray]

    @property
    def nodes(self) -> np.ndarray:
        """Shape ``((n + 1)^2, 2)``: node ``(i, j)`` at flat index ``i (n + 1) + j``."""
        return self.box_grid.nodes

    @property
    def interior(self) -> np.ndarray:
        """Shape ``((n + 1)^2,)``, boolean: the nodes that carry unknowns."""
        return self.box_grid.interior


@dataclass(frozen=True, kw_only=True)
class Monotone17(Scheme):
    """The monotone wide-stencil scheme of Froese and Oberman (2011) on a 2D box.

    At every interior node the discrete equation is

        min over pairs of a^+ b^+ = f,

    with ``a`` and ``b`` the second differences along the two directions of
    an orthogonal pair of the stencil (:func:`stencil_pairs`). The scheme is
    monotone and its solutions converge to the viscosity solution; its
    accuracy is limited by the angular resolution of the stencil as well as
    by ``h``. Below zero each pair's value continues with the slope
    ``kappa = sqrt(max(f, mean f, 1))`` (:func:`pair_values`). That leaves
    the value of every discretely convex grid function unchanged, and a root
    has no negative second difference in any pair that enters the minimum;
    without it, where ``f = 0`` any grid function with a negative second
    difference in every pair solves the equation. As ``kappa >= 1``, a
    negative second difference ``a`` adds at least ``|a|`` to the residual,
    so ``Newton``'s stopping rule cannot accept a non-convex iterate that
    is not a root.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    stencil
        Points of the stencil: 9, 17 or 33.
    boundary
        What a node does with a direction whose arm leaves the box.
        ``"narrow"`` leaves the pair out of the minimum there, so the
        stencil narrows towards the boundary (the axis pair is always
        complete). ``"interpolate"`` shortens the arm to the boundary and
        takes the Dirichlet data there (Oberman 2008, Section 3.1). The
        papers describe the second; their tables match the first
        (``benchmarks/monotone17.md``, Finding 1).
    jacobian_floor
        Positive; :meth:`jacobian` keeps every factor of a product at least
        ``jacobian_floor / (2 h^2)``. Froese and Oberman (2011, Section 5.3)
        take ``1e-8``; that floor makes Newton converge only linearly where
        ``f = 0`` (hundreds of steps on T20 at ``n = 181``), and the default
        ``1e-12`` does not. The floor changes the steps, not the solution.
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
        ``(2,)``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2,)

    n: int = 64
    stencil: Literal[9, 17, 33] = 17
    boundary: Literal["narrow", "interpolate"] = "narrow"
    jacobian_floor: float = 1e-12
    spline_degree: int = 3

    def __post_init__(self) -> None:
        n = integer("n", self.n, 2)
        stencil = integer("stencil", self.stencil, 1)
        stencil_pairs(stencil)  # raises for a size other than 9, 17 and 33
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
        object.__setattr__(self, "stencil", cast(Literal[9, 17, 33], stencil))
        object.__setattr__(
            self, "jacobian_floor", positive("jacobian_floor", self.jacobian_floor)
        )
        object.__setattr__(self, "spline_degree", spline_degree)

    @property
    def angular_resolution(self) -> float:
        """The stencil's angular resolution ``dtheta`` (:func:`angular_resolution`)."""
        return angular_resolution(self.stencil)

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is a 2D box with equal sides and a Dirichlet boundary.

        A pair of orthogonal index directions is orthogonal in space only on
        square cells, hence the equal sides.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        if not (
            isinstance(problem.domain, self.domains)
            and isinstance(problem.boundary, self.boundaries)
            and problem.dim in self.dims
        ):
            return False
        box = problem.domain.bbox
        width, height = np.subtract(box.hi, box.lo)
        return bool(np.isclose(width, height, rtol=1e-12, atol=0.0))

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
            raise UnsupportedProblemError(
                f"Monotone17 does not support {problem.name}: it needs a 2D Box "
                "with equal sides and a Dirichlet boundary"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        n, box, g = self.n, problem.domain.bbox, problem.boundary.g
        box_grid = BoxGrid(box, n)
        (x, y), nodes, interior = box_grid.axes, box_grid.nodes, box_grid.interior
        pairs = stencil_pairs(self.stencil)
        parts = [
            directional_difference(x, y, (int(a), int(b)), g)
            for a, b in pairs.reshape(-1, 2)
        ]
        operators = [part[0] for part in parts]
        shape = (len(pairs), 2, (n - 1) ** 2)
        full = np.array([part[2] for part in parts]).reshape(shape)
        if self.boundary == "narrow":
            available = full[:, 0] & full[:, 1]  # both directions of a pair
        else:
            available = np.ones((len(pairs), (n - 1) ** 2), dtype=bool)
        f = problem.f(nodes[interior])
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError(
                f"Monotone17 needs a finite f >= 0 at the interior nodes, and "
                f"{problem.name} has f from {np.min(f)} to {np.max(f)} there"
            )
        h = box_grid.h[0]
        floor = self.jacobian_floor / (2.0 * h * h)
        return MonotoneGrid(
            box_grid=box_grid,
            f=f,
            boundary_values=g(nodes[~interior]),
            pairs=len(pairs),
            operators=sp.csr_matrix(sp.vstack(operators)),
            data=np.array([part[1] for part in parts]).reshape(shape),
            available=available,
            kappa=np.sqrt(np.maximum(np.maximum(f, np.mean(f)), 1.0)),
            floor=floor,
            laplace=factorized((operators[0] + operators[1]).tocsc()),
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
        """Return the second differences of every pair at the interior nodes.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        numpy.ndarray
            Shape ``(pairs, 2, m)``, in the order of :func:`stencil_pairs`.
        """
        return (grid.operators @ v).reshape(grid.data.shape) + grid.data

    def residual(self, grid: MonotoneGrid, v: np.ndarray) -> np.ndarray:
        """``min over pairs of pair_values - f`` at the interior nodes.

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
        return _available_values(grid, d).min(axis=0) - grid.f

    def jacobian(self, grid: MonotoneGrid, v: np.ndarray) -> sp.csc_matrix:
        """Return a generalised Jacobian ``J`` of :meth:`residual`; ``-J`` is an M-matrix.

        At each node the pair that attains the minimum is differentiated
        (Danskin's theorem) with the coefficients of
        :func:`pair_jacobian_coefficients`.

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
        active = _available_values(grid, d).argmin(axis=0)
        chosen = np.arange(grid.pairs)[:, None] == active
        weights = pair_jacobian_coefficients(d, grid.kappa, grid.floor)
        weights = (weights * chosen[:, None, :]).ravel()
        rows = np.tile(np.arange(v.size), 2 * grid.pairs)
        collapse = sp.csr_matrix(
            (weights, (rows, np.arange(weights.size))), shape=(v.size, weights.size)
        )
        return sp.csc_matrix(collapse @ grid.operators)

    def solve_laplace(self, grid: MonotoneGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplace u = rhs`` with the Dirichlet data for the unknowns.

        The Laplacian is the sum of the axis pair's differences, the 5-point
        Laplacian.

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
            ``node_hessian`` holds the centred 9-point second differences at
            the interior nodes (the scheme has no Hessian of its own; these
            are the 9-point stencil's axis differences, and ``u_xy`` is half
            the difference of its two diagonal ones) and NaN at the boundary
            nodes. ``info`` holds the spacing ``h`` and ``residual_max``,
            the max absolute residual.
        """
        values = np.empty(len(grid.nodes))
        values[grid.interior] = v
        values[~grid.interior] = grid.boundary_values
        n, h = grid.box_grid.n, grid.box_grid.h[0]
        return Solution(
            # a spline needs degree + 1 knots, so a coarse grid lowers the degree
            u=grid.box_grid.interpolant(values, min(self.spline_degree, n)),
            nodes=grid.nodes,
            values=values,
            node_hessian=centred_hessian(grid.box_grid, values),
            interior=grid.interior,
            unknowns=int(v.size),
            info={
                "h": (h, h),
                "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
            },
        )


def _available_values(grid: MonotoneGrid, d: np.ndarray) -> np.ndarray:
    """:func:`pair_values`, ``+inf`` where a pair is not available."""
    return np.where(grid.available, pair_values(d, grid.kappa), np.inf)
