"""The standard scheme on a box in two or three dimensions.

The centred second differences of :func:`~maesolvers.classical.grid.second_differences`
(9 points in 2D, 19 in 3D) in the Monge-Ampere operator ``det D^2 u``, with
the Dirichlet data on the boundary ring: in 2D the equations of
:class:`~maesolvers.classical.schemes.Standard9` on a box, in 3D those of
Liu, Froese, Oberman and Xiao (Int. J. Comput. Math. 94 (2017), eqs. (4) to
(10)). :class:`StandardBox` evaluates them by slicing the array of node
values, so a residual costs a few passes over the ``(n + 1)^d`` values and
no matrix, and it builds the sparse Jacobian and the Laplacian factorisation
only when an iteration asks for them; this keeps grids of ``8192^2`` nodes,
which the multigrid iteration reaches, within memory.

The kernels are module-level pure functions: :func:`centred_differences`
(the discrete Hessian's entries at the interior nodes),
:func:`determinant` and :func:`cofactors` (of a symmetric matrix given by
its entries) and :func:`smallest_root`, the pointwise solve of the nonlinear
Gauss-Seidel iteration, which picks the root with a positive definite
discrete Hessian (Benamou-Froese-Oberman 2010 in 2D; Liu et al. 2017,
Theorems 2.1 and 2.2, in 3D).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from maesolvers.classical.grid import BoxGrid, second_differences
from maesolvers.classical.schemes import Scheme
from maesolvers.core._params import integer
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError

Entries = Mapping[tuple[int, int], np.ndarray]
"""The entries ``(k, l)``, ``k <= l``, of symmetric ``d x d`` matrices, one array each."""

_NEWTON_STEPS = 100
"""Most Newton steps of :func:`smallest_root` in 3D; a simple root takes a handful."""


def centred_differences(
    u: np.ndarray, h: Sequence[float]
) -> dict[tuple[int, int], np.ndarray]:
    """Return the centred second differences of node values at the interior nodes.

    The differences of :func:`~maesolvers.classical.grid.second_differences`,
    computed by slicing the array of node values: ``D_kk`` the 3-point
    difference along axis ``k``, ``D_kl``, ``k < l``, the 4-point cross
    difference.

    Parameters
    ----------
    u
        Shape ``(n + 1,) * d``, ``n >= 2``: the values at the nodes.
    h
        The spacing along each axis, ``d`` numbers.

    Returns
    -------
    dict
        ``D_kl`` for ``k <= l`` under the key ``(k, l)``, each of shape
        ``(n - 1,) * d``: at the interior nodes, in the grid's C order.
    """
    u = np.asarray(u, dtype=float)
    d = u.ndim
    if len(h) != d:
        raise ValueError(f"h must hold {d} spacings, got {len(h)}")

    def shifted(step: dict[int, int]) -> np.ndarray:
        index = tuple(
            slice(1 + step.get(k, 0), u.shape[k] - 1 + step.get(k, 0)) for k in range(d)
        )
        return u[index]

    centre = shifted({})
    out = {}
    for k in range(d):
        out[k, k] = (shifted({k: 1}) - 2.0 * centre + shifted({k: -1})) / h[k] ** 2
        for m in range(k + 1, d):
            out[k, m] = (
                shifted({k: 1, m: 1})
                - shifted({k: 1, m: -1})
                - shifted({k: -1, m: 1})
                + shifted({k: -1, m: -1})
            ) / (4.0 * h[k] * h[m])
    return out


def determinant(a: Entries, d: int) -> np.ndarray:
    """Return the determinant of symmetric ``d x d`` matrices given by their entries.

    Parameters
    ----------
    a
        The entries ``(k, l)``, ``k <= l``, arrays of one shape.
    d
        2 or 3.

    Returns
    -------
    numpy.ndarray
        The determinants, of the entries' shape.
    """
    if d == 2:
        return a[0, 0] * a[1, 1] - a[0, 1] ** 2
    if d == 3:
        return (
            a[0, 0] * a[1, 1] * a[2, 2]
            + 2.0 * a[0, 1] * a[0, 2] * a[1, 2]
            - a[0, 0] * a[1, 2] ** 2
            - a[1, 1] * a[0, 2] ** 2
            - a[2, 2] * a[0, 1] ** 2
        )
    raise ValueError(f"d must be 2 or 3, got {d}")


def cofactors(a: Entries, d: int) -> dict[tuple[int, int], np.ndarray]:
    """Return the cofactors of symmetric ``d x d`` matrices given by their entries.

    The cofactor ``C_kl`` is the derivative of :func:`determinant` with
    respect to the entry ``a_kl`` of a general matrix, so the derivative of
    the determinant of a symmetric matrix along a symmetric perturbation
    ``E`` is ``sum_k C_kk E_kk + 2 sum_{k<l} C_kl E_kl``.

    Parameters
    ----------
    a
        The entries ``(k, l)``, ``k <= l``, arrays of one shape.
    d
        2 or 3.

    Returns
    -------
    dict
        ``C_kl`` for ``k <= l`` under the key ``(k, l)``.
    """
    if d == 2:
        return {(0, 0): a[1, 1], (1, 1): a[0, 0], (0, 1): -a[0, 1]}
    if d == 3:
        return {
            (0, 0): a[1, 1] * a[2, 2] - a[1, 2] ** 2,
            (1, 1): a[0, 0] * a[2, 2] - a[0, 2] ** 2,
            (2, 2): a[0, 0] * a[1, 1] - a[0, 1] ** 2,
            (0, 1): a[0, 2] * a[1, 2] - a[0, 1] * a[2, 2],
            (0, 2): a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1],
            (1, 2): a[0, 1] * a[0, 2] - a[0, 0] * a[1, 2],
        }
    raise ValueError(f"d must be 2 or 3, got {d}")


def smallest_root(
    a: Entries, c: np.ndarray, d: int, start: np.ndarray | None = None
) -> np.ndarray:
    """Return the smallest real root ``t`` of ``det(A - t I) = c``.

    The root selection of the nonlinear Gauss-Seidel iteration
    (Benamou-Froese-Oberman 2010 in 2D; Liu et al. 2017, eq. (12), in 3D).
    For ``c > 0`` the smallest real root lies below every eigenvalue of
    ``A``, and it is the only root at which ``A - t I`` is positive
    definite: ``det(A - t I) = prod_k (mu_k - t)`` falls from infinity to 0
    as ``t`` rises to the smallest eigenvalue ``mu_1`` (Liu et al. 2017,
    Theorems 2.1 and 2.2). For ``c <= 0``, which a coarse grid of a
    multigrid cycle can ask for near a singularity, the smallest real root
    lies at or above ``mu_1``, and ``A - t I`` is not positive definite
    there; in 2D, where there may be no real root, the result is then the
    vertex ``(a_00 + a_11) / 2``.

    In 2D the root is the closed form
    ``(a_00 + a_11) / 2 - sqrt(((a_00 - a_11) / 2)^2 + a_01^2 + c)``
    (Benamou-Froese-Oberman 2010), the square root taken of 0 where its
    argument is negative. In 3D, where ``c > 0``, it is Newton's method on
    ``det(A - t I) - c``, which is decreasing and convex below ``mu_1``, so
    from a start below the root the iterates rise to it monotonically: the
    start is ``start`` where ``A - start I`` is positive definite with
    determinant at least ``c``, else Gershgorin's lower bound for ``mu_1``
    less ``c^(1/3)``. Where ``c <= 0`` it is the smallest real root of the
    cubic's closed form, refined by Newton steps that reduce the residual.

    Parameters
    ----------
    a
        The entries ``(k, l)``, ``k <= l``, of symmetric matrices ``A``,
        arrays of shape ``(m,)``.
    c
        Shape ``(m,)``: the right-hand sides.
    d
        2 or 3.
    start
        Shape ``(m,)``, or None: a guess for the roots, such as the
        current iterate of a Gauss-Seidel sweep (used in 3D only).

    Returns
    -------
    numpy.ndarray
        Shape ``(m,)``: the roots.
    """
    c = np.asarray(c, dtype=float)
    if d == 2:
        mean = 0.5 * (a[0, 0] + a[1, 1])
        half_gap = 0.5 * (a[0, 0] - a[1, 1])
        return mean - np.sqrt(np.maximum(half_gap**2 + a[0, 1] ** 2 + c, 0.0))
    if d != 3:
        raise ValueError(f"d must be 2 or 3, got {d}")
    t = np.empty_like(c)
    positive = c > 0.0
    for mask, solve in ((positive, _root_below_spectrum), (~positive, _cubic_root)):
        if np.any(mask):
            sub = {key: value[mask] for key, value in a.items()}
            guess = None if start is None else np.asarray(start, dtype=float)[mask]
            t[mask] = solve(sub, c[mask], guess)
    return t


def _root_below_spectrum(
    a: Entries, c: np.ndarray, start: np.ndarray | None
) -> np.ndarray:
    """Solve the 3D case of :func:`smallest_root` for ``c > 0`` by monotone Newton steps."""
    off = {(k, m): np.abs(a[k, m]) for k in range(3) for m in range(k + 1, 3)}
    gershgorin = np.minimum.reduce(
        [
            a[k, k] - sum(off[min(k, m), max(k, m)] for m in range(3) if m != k)
            for k in range(3)
        ]
    )
    t = gershgorin - np.cbrt(c)
    if start is not None:
        shifted = _shift(a, start)
        below = (
            (shifted[0, 0] > 0.0)
            & (shifted[0, 0] * shifted[1, 1] - shifted[0, 1] ** 2 > 0.0)
            & (determinant(shifted, 3) >= c)
        )
        t = np.where(below, np.maximum(t, start), t)
    active = np.arange(t.size)
    for _ in range(_NEWTON_STEPS):
        value, slope = _cubic_and_slope({k: v[active] for k, v in a.items()}, t[active])
        step = -(value - c[active]) / slope
        moving = step > 4.0 * np.finfo(float).eps * np.maximum(np.abs(t[active]), 1.0)
        t[active[moving]] += step[moving]
        active = active[moving]
        if active.size == 0:
            break
    return t


def _cubic_root(a: Entries, c: np.ndarray, start: np.ndarray | None) -> np.ndarray:
    """Solve the 3D case of :func:`smallest_root` for ``c <= 0`` by the closed form.

    ``det(A - t I) = c`` is ``t^3 - p1 t^2 + p2 t - (p3 - c) = 0`` with the
    trace ``p1``, the sum of the principal 2 x 2 minors ``p2`` and the
    determinant ``p3`` of ``A``; with ``t = s + p1 / 3`` it is
    ``s^3 + P s + Q = 0``, solved by Cardano's formula where it has one real
    root and by the trigonometric form where it has three.
    """
    p1 = a[0, 0] + a[1, 1] + a[2, 2]
    p2 = _trace(cofactors(a, 3), 3)
    p3 = determinant(a, 3)
    big_p = p2 - p1**2 / 3.0
    big_q = -2.0 * p1**3 / 27.0 + p1 * p2 / 3.0 - (p3 - c)
    gap = (big_q / 2.0) ** 2 + (big_p / 3.0) ** 3
    one = gap >= 0.0
    s = np.empty_like(c)
    root = np.sqrt(gap[one])
    s[one] = np.cbrt(-big_q[one] / 2.0 + root) + np.cbrt(-big_q[one] / 2.0 - root)
    p, q = big_p[~one], big_q[~one]  # p < 0 here
    radius = 2.0 * np.sqrt(-p / 3.0)
    angle = np.arccos(np.clip(1.5 * q / p * np.sqrt(-3.0 / p), -1.0, 1.0)) / 3.0
    s[~one] = np.min(
        [radius * np.cos(angle - 2.0 * np.pi * k / 3.0) for k in range(3)], axis=0
    )
    t = s + p1 / 3.0
    for _ in range(3):  # Newton refinement, kept where it reduces the residual
        value, slope = _cubic_and_slope(a, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            refined = t - (value - c) / slope
        new_value, _ = _cubic_and_slope(a, refined)
        better = np.isfinite(refined) & (np.abs(new_value - c) < np.abs(value - c))
        t = np.where(better, refined, t)
    return t


def _cubic_and_slope(a: Entries, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``det(A - t I)`` and its derivative in ``t``, from the shifted entries."""
    shifted = _shift(a, t)
    slope = -_trace(cofactors(shifted, 3), 3)
    return determinant(shifted, 3), slope


def _trace(a: Mapping[tuple[int, int], Any], d: int) -> Any:
    """Return the sum of the diagonal entries ``a[k, k]``, arrays or matrices."""
    total = a[0, 0]
    for k in range(1, d):
        total = total + a[k, k]
    return total


def _shift(a: Entries, t: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Return the entries of ``A - t I``."""
    return {(k, m): value - t if k == m else value for (k, m), value in a.items()}


@dataclass(frozen=True, eq=False)
class BoxContext:
    """One problem discretised by :class:`StandardBox`, its working context.

    Only :class:`StandardBox` reads the fields; they are not a stable
    interface and may change with the scheme. The unknowns are the values at
    the interior nodes of :attr:`grid`, in flat order; the boundary ring
    carries the Dirichlet data.

    Attributes
    ----------
    grid
        The grid of ``n`` cells per side over the box.
    f
        The right-hand side at the unknowns.
    boundary_values
        The Dirichlet data on the boundary ring, in flat order.
    spline_degree
        The degree of the tensor spline through the node values.
    """

    grid: BoxGrid
    f: np.ndarray
    boundary_values: np.ndarray
    spline_degree: int

    @cached_property
    def operators(
        self,
    ) -> tuple[
        dict[tuple[int, int], sp.csc_matrix], dict[tuple[int, int], sp.csc_matrix]
    ]:
        """The second differences split into the columns of the unknowns and of the ring.

        Built on first use: only :meth:`StandardBox.jacobian` and
        :meth:`StandardBox.solve_laplace` need matrices.
        """
        interior = self.grid.interior
        full = second_differences(self.grid.n, self.grid.h)
        return (
            {key: op[:, interior].tocsc() for key, op in full.items()},
            {key: op[:, ~interior].tocsc() for key, op in full.items()},
        )

    @cached_property
    def laplace(self) -> Callable[[np.ndarray], np.ndarray]:
        """Solves ``sum_k D_kk v = rhs`` on the unknowns by a sparse LU factorisation."""
        inner, _ = self.operators
        return factorized(_trace(inner, self.grid.dim).tocsc())

    def values(self, v: np.ndarray) -> np.ndarray:
        """Return the values at every node: the unknowns ``v`` and the Dirichlet data."""
        values = np.empty(len(self.grid.interior))
        values[self.grid.interior] = v
        values[~self.grid.interior] = self.boundary_values
        return values

    def hessian(self, values: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        """:func:`centred_differences` of node values, flattened to the unknowns' order."""
        out = centred_differences(values.reshape(self.grid.shape), self.grid.h)
        return {key: value.ravel() for key, value in out.items()}


@dataclass(frozen=True, kw_only=True)
class StandardBox(Scheme):
    """The standard scheme on a box in 2D or 3D: centred differences, ``det D^2 u = f``.

    The discrete Hessian is the centred differences at each interior node,
    9 points in 2D and 19 in 3D, and the equation is its determinant equal
    to ``f``, with the Dirichlet data on the boundary ring (Liu et al. 2017,
    eqs. (4) to (10)); in 2D these are the equations of
    :class:`~maesolvers.classical.schemes.Standard9` on a box. The scheme is
    consistent but not monotone: its equations have non-convex roots as well,
    and the iteration decides which one a solve reaches.

    It provides ``residual``, ``jacobian`` and ``solve_laplace``, so
    :class:`~maesolvers.classical.iterations.Newton` solves it, and
    :meth:`relax`, the pointwise solve that
    :class:`~maesolvers.classical.multigrid.Multigrid` sweeps with.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit box, ADR 0001), at least 2;
        ``(n + 1)^d`` nodes and ``(n - 1)^d`` unknowns.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``.

    Attributes
    ----------
    domains : tuple of type
        ``(Box,)``.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2, 3)``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2, 3)

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
        """Whether the problem is a 2D or 3D box with a Dirichlet boundary.

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

    def discretize(self, problem: Problem) -> BoxContext:
        """Build the grid with ``n`` cells per side over the box.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        BoxContext

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        """
        if not self.supports(problem):
            raise UnsupportedProblemError(
                f"StandardBox does not support {problem.name}: it needs a 2D or 3D "
                "Box with a Dirichlet boundary"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        assert isinstance(problem.domain, Box)
        grid = BoxGrid(problem.domain, self.n)
        return BoxContext(
            grid=grid,
            f=problem.f(grid.nodes[grid.interior]),
            boundary_values=problem.boundary.g(grid.nodes[~grid.interior]),
            spline_degree=min(self.spline_degree, self.n),  # degree + 1 knots
        )

    def rhs(self, grid: BoxContext) -> np.ndarray:
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

    def residual(self, grid: BoxContext, v: np.ndarray) -> np.ndarray:
        """``det D^2 u - f`` at the unknowns.

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
        hess = grid.hessian(grid.values(v))
        return determinant(hess, grid.grid.dim) - grid.f

    def jacobian(self, grid: BoxContext, v: np.ndarray) -> sp.csc_matrix:
        """Jacobian of :meth:`residual` with respect to the unknowns.

        ``sum_k diag(C_kk) D_kk + 2 sum_{k<l} diag(C_kl) D_kl``, with the
        :func:`cofactors` ``C`` of the discrete Hessian and the second
        differences ``D`` restricted to the unknowns.

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
        inner, _ = grid.operators
        cofactor = cofactors(grid.hessian(grid.values(v)), grid.grid.dim)
        jacobian = sp.csc_matrix(inner[0, 0].shape)
        for (k, m), cof in cofactor.items():
            jacobian = jacobian + sp.diags((1.0 if k == m else 2.0) * cof) @ inner[k, m]
        return jacobian.tocsc()

    def solve_laplace(self, grid: BoxContext, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplace u = rhs`` with the Dirichlet data for the unknowns.

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
        _, ring = grid.operators
        lap_b = _trace(ring, grid.grid.dim) @ grid.boundary_values
        return grid.laplace(rhs - lap_b)

    def relax(
        self, grid: BoxContext, u: np.ndarray, b: np.ndarray, nodes: np.ndarray
    ) -> np.ndarray:
        """Solve the equation of each node for its own value, the others held.

        The nonlinear Gauss-Seidel update: at each node of ``nodes`` the
        value ``v`` for which the discrete Hessian, its neighbours' values
        held at ``u``, has determinant ``b`` and is positive definite. With
        ``A_kk = u(i + e_k) + u(i - e_k)`` and ``A_kl = h_k h_l D_kl u``, the
        Hessian is ``H^-1 (A - 2 v I) H^-1``, ``H = diag(h)``, so ``2 v`` is
        the :func:`smallest_root` of ``det(A - t I) = b prod_k h_k^2``. The
        nodes are updated together, each from the values in ``u``, so a
        caller sweeps in Gauss-Seidel order by passing nodes that are not
        each other's neighbours.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        u
            Shape ``((n + 1)^d,)``: the values at every node, in flat order.
        b
            The right-hand side at each node of ``nodes``; ``f`` for the
            scheme's own equations, another one on a coarse grid of a
            multigrid cycle.
        nodes
            Flat indices of interior nodes.

        Returns
        -------
        numpy.ndarray
            The new values at ``nodes``.
        """
        box, d = grid.grid, grid.grid.dim
        steps = [box.offset([int(j == k) for j in range(d)]) for k in range(d)]
        a: dict[tuple[int, int], np.ndarray] = {}
        for k in range(d):
            a[k, k] = u[nodes + steps[k]] + u[nodes - steps[k]]
            for m in range(k + 1, d):
                plus, minus = steps[k] + steps[m], steps[k] - steps[m]
                a[k, m] = 0.25 * (
                    u[nodes + plus]
                    - u[nodes + minus]
                    - u[nodes - minus]
                    + u[nodes - plus]
                )
        scale = float(np.prod(np.square(box.h)))
        return 0.5 * smallest_root(a, scale * np.asarray(b), d, start=2.0 * u[nodes])

    def solution(self, grid: BoxContext, v: np.ndarray) -> Solution:
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
            ``u`` is the tensor spline through the node values. ``info``
            holds the spacing ``h``; ``residual_max``, the max absolute value
            of :meth:`residual`; and ``interpolant``, ``"spline(k)"``.
        """
        box, d = grid.grid, grid.grid.dim
        values = grid.values(v)
        hess = grid.hessian(values)
        node_hessian = np.full((len(values), d, d), np.nan)
        for (k, m), entry in hess.items():
            node_hessian[box.interior, k, m] = node_hessian[box.interior, m, k] = entry
        residual = determinant(hess, d) - grid.f
        return Solution(
            u=box.interpolant(values, grid.spline_degree),
            nodes=box.nodes,
            values=values,
            node_hessian=node_hessian,
            interior=box.interior,
            unknowns=int(v.size),
            info={
                "h": box.h,
                "residual_max": float(np.max(np.abs(residual))),
                "interpolant": f"spline({grid.spline_degree})",
            },
        )
