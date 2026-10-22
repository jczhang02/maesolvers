"""The hybrid scheme of Froese and Oberman (J. Comput. Phys. 2011).

The hybrid scheme (Froese-Oberman, J. Comput. Phys. 230 (2011), Section 5)
is a weighted average of two schemes on one grid,

    MA_H[u] = w MA_N[u] + (1 - w) MA_M[u] = f,

the standard 9-point scheme ``MA_N`` (:class:`~maesolvers.classical.schemes.Standard9`),
accurate where the solution is smooth, and the monotone wide-stencil scheme
``MA_M`` (:class:`~maesolvers.classical.monotone.Monotone17`), which
converges to the viscosity solution, near the places where the solution may
be singular. The weight ``w`` is fixed before the solve from the data alone:
the switching rule :func:`hybrid_weights` sets it to 0 near a possible
singular set and to 1 elsewhere.

:class:`Hybrid` composes the two schemes and the rule into a
:class:`~maesolvers.classical.schemes.Scheme` that
:class:`~maesolvers.classical.iterations.Newton` solves, from the root of the
monotone scheme as the paper does (its Section 7.1); a rule of your own is a
subclass that overrides :meth:`Hybrid.weights`.
"""

# No `from __future__ import annotations` here: jsonargparse reads the field
# types (ADR 0005); see maesolvers.classical.iterations.

import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt

from maesolvers.classical.monotone import Monotone17, MonotoneGrid
from maesolvers.classical.schemes import Grid, Scheme, Standard9
from maesolvers.core._params import nonnegative, positive
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError

_ON_THE_BAND = 1e-9
"""Relative slack of the band edge, so a node exactly ``band`` steps away is in it."""


def hybrid_weights(f: np.ndarray, epsilon: float, band: float) -> np.ndarray:
    """Return the weights of the standard scheme by Froese-Oberman's switching rule.

    The possible singular set ``Omega^s`` (Froese-Oberman 2011, eq. (6)) is
    where ``f`` is at most ``epsilon`` or at least ``1 / epsilon`` (or not
    finite), together with the boundary: on a box every side is flat, and
    the regularity results (their conditions (2)) need a strictly convex
    domain. The weight is 0 at every node within ``band`` grid steps of a
    node of ``Omega^s`` (Euclidean distance between nodes) and 1 at the
    other nodes.

    Parameters
    ----------
    f
        Shape ``(n - 1, n - 1)``: the right-hand side at the interior nodes of
        a grid of ``n`` by ``n`` square cells, node ``(i, j)`` at
        ``f[i - 1, j - 1]``.
    epsilon
        Positive; the thresholds are ``epsilon`` and ``1 / epsilon``.
    band
        Nonnegative, in grid steps: the radius of the neighbourhood of
        ``Omega^s`` where the weight is 0. With ``band < 1`` the boundary
        alone sets no weight to 0.

    Returns
    -------
    numpy.ndarray
        Shape ``(n - 1, n - 1)``, each entry 0 or 1.
    """
    f = np.asarray(f, dtype=float)
    singular = np.ones((f.shape[0] + 2, f.shape[1] + 2), dtype=bool)
    with np.errstate(invalid="ignore"):
        regular = (f > epsilon) & (f < 1.0 / epsilon)
    singular[1:-1, 1:-1] = ~regular
    distance = np.asarray(distance_transform_edt(~singular))[1:-1, 1:-1]
    return np.where(distance <= band * (1.0 + _ON_THE_BAND), 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class HybridGrid:
    """One problem discretised by :class:`Hybrid`, its working context.

    Only :class:`Hybrid` reads the fields; they are not a stable interface.

    Attributes
    ----------
    standard
        The grid of :attr:`Hybrid.standard`.
    monotone
        The grid of :attr:`Hybrid.monotone`, with the same nodes and unknowns.
    weight
        ``w`` at the unknowns, each in ``[0, 1]``.
    """

    standard: Grid
    monotone: MonotoneGrid
    weight: np.ndarray


@dataclass(frozen=True, kw_only=True)
class Hybrid(Scheme):
    """The hybrid scheme of Froese and Oberman (2011) on a 2D box.

    At every interior node the discrete equation is

        w MA_N[u] + (1 - w) MA_M[u] = f,

    with ``MA_N`` the standard 9-point scheme (:class:`Standard9`), ``MA_M``
    the monotone wide-stencil scheme (:class:`Monotone17`) and the weight
    ``w`` from :meth:`weights`, fixed before the solve. By default ``w`` is
    0 within ``band`` grid steps of the possible singular set, where ``f``
    is at most ``epsilon`` or at least ``1 / epsilon`` or the boundary is
    flat (every side of a box), and 1 elsewhere (:func:`hybrid_weights`).
    The residual and the Jacobian are the same averages of the two schemes'.
    The standard part has non-convex roots: ``Newton`` from its Poisson
    start reaches one on T2 from ``n = 63`` on, and from the root of
    ``Monotone17`` (``start=``, as :class:`FDHybrid
    <maesolvers.classical.presets.FDHybrid>` does) the convex one
    (``benchmarks/hybrid.md``).

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    stencil
        Points of the monotone scheme's stencil: 9, 17 (the paper's) or 33.
    boundary
        ``"narrow"`` or ``"interpolate"``: what the monotone scheme does with
        a direction whose arm leaves the box, as in :class:`Monotone17`.
    jacobian_floor
        The floor of the monotone Jacobian's factors, times ``1 / (2 h^2)``,
        as in :class:`Monotone17`.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5 and at most ``n``.
    epsilon
        The threshold of the switching rule, positive, or None for the
        paper's choice ``h``.
    band
        Nonnegative: the radius, in grid steps, of the neighbourhood of the
        possible singular set where ``w = 0``; the paper's ``h``
        neighbourhood is 1.

    Attributes
    ----------
    standard : Standard9
        The standard scheme on the same grid, built from ``n`` and
        ``spline_degree``.
    monotone : Monotone17
        The monotone scheme on the same grid, built from the other fields.
    domains : tuple of type
        ``(Box,)``; :meth:`supports` also asks for equal sides.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2,)``.

    Raises
    ------
    ValueError
        On construction, if a parameter is out of range.
    """

    domains: ClassVar[tuple[type, ...]] = Monotone17.domains
    boundaries: ClassVar[tuple[type, ...]] = Monotone17.boundaries
    dims: ClassVar[tuple[int, ...]] = Monotone17.dims

    standard: Standard9 = field(init=False, repr=False)
    monotone: Monotone17 = field(init=False, repr=False)
    n: int = Monotone17.n
    stencil: Literal[9, 17, 33] = Monotone17.stencil
    boundary: Literal["narrow", "interpolate"] = Monotone17.boundary
    jacobian_floor: float = Monotone17.jacobian_floor
    spline_degree: int = Monotone17.spline_degree
    epsilon: float | None = None
    band: float = 1.0

    def __post_init__(self) -> None:
        monotone = Monotone17(
            n=self.n,
            stencil=self.stencil,
            boundary=self.boundary,
            jacobian_floor=self.jacobian_floor,
            spline_degree=self.spline_degree,
        )
        standard = Standard9(n=monotone.n, spline_degree=monotone.spline_degree)
        object.__setattr__(self, "monotone", monotone)
        object.__setattr__(self, "standard", standard)
        object.__setattr__(self, "n", monotone.n)
        object.__setattr__(self, "stencil", monotone.stencil)
        object.__setattr__(self, "jacobian_floor", monotone.jacobian_floor)
        object.__setattr__(self, "spline_degree", monotone.spline_degree)
        if self.epsilon is not None:
            object.__setattr__(self, "epsilon", positive("epsilon", self.epsilon))
        object.__setattr__(self, "band", nonnegative("band", self.band))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is a 2D box with equal sides and a Dirichlet boundary.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
            What :meth:`Monotone17.supports` returns.
        """
        return self.monotone.supports(problem)

    def weights(self, problem: Problem) -> np.ndarray:
        """Return the weight ``w`` of the standard scheme at the unknowns.

        The plug-in point for a switching rule of your own: override it in a
        subclass. This one is Froese and Oberman's rule,
        :func:`hybrid_weights` with ``epsilon`` (``h`` if None) and ``band``.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        numpy.ndarray
            Shape ``((n - 1)^2,)``, in ``[0, 1]``, node ``(i, j)`` of the
            ``n`` by ``n`` cells over the box at ``(i - 1) (n - 1) + j - 1``.
        """
        box, n = problem.domain.bbox, self.n
        x = np.linspace(box.lo[0], box.hi[0], n + 1)[1:-1]
        y = np.linspace(box.lo[1], box.hi[1], n + 1)[1:-1]
        nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
        h = (box.hi[0] - box.lo[0]) / n
        epsilon = h if self.epsilon is None else self.epsilon
        return hybrid_weights(problem.f(nodes), epsilon, self.band).ravel()

    def discretize(self, problem: Problem) -> HybridGrid:
        """Build both schemes' grids and the weights.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        HybridGrid

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        ValueError
            If ``f`` is negative or not finite at an interior node (from
            :meth:`Monotone17.discretize`), or :meth:`weights` returns the
            wrong shape or a value outside ``[0, 1]``.
        """
        if not self.supports(problem):
            raise UnsupportedProblemError(
                f"Hybrid does not support {problem.name}: it needs a 2D Box "
                "with equal sides and a Dirichlet boundary"
            )
        standard = self.standard.discretize(problem)
        monotone = self.monotone.discretize(problem)
        # Both schemes number the nodes and the unknowns alike on a box.
        assert np.array_equal(standard.interior, monotone.interior)
        weight = np.asarray(self.weights(problem), dtype=float)
        unknowns = monotone.f.size
        if weight.shape != (unknowns,):
            raise ValueError(
                f"{type(self).__name__}.weights must return shape ({unknowns},), "
                f"got {weight.shape}"
            )
        if not np.all((weight >= 0.0) & (weight <= 1.0)):
            raise ValueError(
                f"{type(self).__name__}.weights must lie in [0, 1], got values "
                f"from {np.min(weight)} to {np.max(weight)}"
            )
        weight.setflags(write=False)
        return HybridGrid(standard=standard, monotone=monotone, weight=weight)

    def rhs(self, grid: HybridGrid) -> np.ndarray:
        """Return the right-hand side ``f`` at the unknowns.

        Parameters
        ----------
        grid
            From :meth:`discretize`.

        Returns
        -------
        numpy.ndarray
        """
        return grid.monotone.f

    def residual(self, grid: HybridGrid, v: np.ndarray) -> np.ndarray:
        """``w (MA_N[u] - f) + (1 - w) (MA_M[u] - f)`` at the interior nodes.

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
        w = grid.weight
        return w * self.standard.residual(grid.standard, v) + (
            1.0 - w
        ) * self.monotone.residual(grid.monotone, v)

    def jacobian(self, grid: HybridGrid, v: np.ndarray) -> sp.csc_matrix:
        """Return the same average of the two schemes' (generalised) Jacobians.

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
        w = grid.weight
        standard = self.standard.jacobian(grid.standard, v)
        monotone = self.monotone.jacobian(grid.monotone, v)
        return sp.csc_matrix(sp.diags(w) @ standard + sp.diags(1.0 - w) @ monotone)

    def solve_laplace(self, grid: HybridGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplace u = rhs`` with the Dirichlet data for the unknowns.

        Both schemes' Laplacian is the 5-point one; this is the standard
        scheme's.

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
        return self.standard.solve_laplace(grid.standard, rhs)

    def solution(self, grid: HybridGrid, v: np.ndarray) -> Solution:
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
            The standard scheme's :meth:`Standard9.solution`, with
            ``residual_max`` the max absolute value of :meth:`residual`, and
            ``monotone_fraction``, the share of unknowns where ``w < 1``.
        """
        solution = self.standard.solution(grid.standard, v)
        info = {
            **solution.info,
            "residual_max": float(np.max(np.abs(self.residual(grid, v)))),
            "monotone_fraction": float(np.mean(grid.weight < 1.0)),
        }
        return dataclasses.replace(solution, info=info)
