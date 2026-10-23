"""The full multigrid iteration of Liu, Froese, Oberman and Xiao (2017).

The nonlinear full-approximation-storage (FAS) multigrid method with full
multigrid (FMG) initialisation, smoothed by a nonlinear Gauss-Seidel sweep
that picks the convex root at each node (J. Liu, B. D. Froese, A. M.
Oberman, M. Xiao, Int. J. Comput. Math. 94 (2017) 1850-1866, Algorithms 1
and 2). :class:`Multigrid` is an iteration (ADR 0004) for any scheme on a box
whose unknowns are the interior nodes of a
:class:`~maesolvers.classical.grid.BoxGrid` and that provides the node
values ``values`` and the pointwise solve ``relax``, such as
:class:`~maesolvers.classical.standard_box.StandardBox`.

The grid transfers and the sweep orders are module-level pure functions:
:func:`sweep_groups` (red-black or lexicographic), :func:`resample` (the
linear interpolation between two grids of the same box, which is injection
onto a nested coarse grid and the (bi/tri)linear prolongation from it),
:func:`half_weighting` (the residual's restriction stencil) and
:func:`cubic_resample` (the FMG interpolation).
"""

# No `from __future__ import annotations` here: jsonargparse reads the field
# types of the iteration and its preset (see the note in iterations.py).

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, cast

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import make_interp_spline

from maesolvers.classical.grid import BoxGrid
from maesolvers.classical.iterations import SchemeIteration
from maesolvers.core._params import integer
from maesolvers.core.domains import Box
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution

Ordering = Literal["lexicographic", "red-black"]
"""The order in which a Gauss-Seidel sweep visits the interior nodes."""


def sweep_groups(n: int, d: int, ordering: Ordering) -> tuple[np.ndarray, ...]:
    """Return the interior nodes of a grid in groups, in the order a sweep updates them.

    The nodes of one group are updated together, each from the values the
    groups before it left.

    ``"lexicographic"`` gives the result of updating the nodes one by one in
    flat order (the last axis fastest), exactly: the group of the node with
    multi-index ``i`` is ``sum_k (d - k) i_k``, and for the stencils of width
    1 with at most two non-zero offsets (the 9-point and 19-point centred
    stencils) a neighbour earlier in flat order has a smaller group, a later
    one a larger group, and none the same. So within a group no node reads
    another, and every node reads its earlier neighbours updated and its
    later ones not, as in the sequential sweep.

    ``"red-black"`` gives two groups, the nodes with an even index sum
    first, then the odd ones. The diagonal neighbours of a node have its
    colour, so a colour's update is a Jacobi step among them, as in a
    vectorised red-black sweep of a 9- or 19-point stencil (Liu et al. 2017,
    Section 4.1).

    Parameters
    ----------
    n
        Cells per side, at least 2.
    d
        Dimension, at least 1.
    ordering
        ``"lexicographic"`` or ``"red-black"``.

    Returns
    -------
    tuple of numpy.ndarray
        The flat indices of the interior nodes of each group, in update
        order; together each interior node once.
    """
    n, d = integer("n", n, 2), integer("d", d, 1)
    if ordering not in ("lexicographic", "red-black"):
        raise ValueError(
            f"ordering must be 'lexicographic' or 'red-black', got {ordering!r}"
        )
    index = np.indices((n - 1,) * d).reshape(d, -1) + 1  # interior, in flat order
    flat = np.ravel_multi_index(tuple(index), (n + 1,) * d)
    if ordering == "red-black":
        colour = index.sum(axis=0) % 2
        return flat[colour == 0], flat[colour == 1]
    group = (np.arange(d, 0, -1)[:, None] * index).sum(axis=0)
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.diff(group[order])) + 1
    return tuple(np.split(flat[order], bounds))


def interpolation_matrix(n_from: int, n_to: int) -> sp.csr_matrix:
    """Linear interpolation between two uniform grids of one interval, as a matrix.

    Parameters
    ----------
    n_from, n_to
        Cells of the grid of the values and of the grid of the result, each
        at least 1.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_to + 1, n_from + 1)``: row ``j`` holds the weights of the
        node values at the point ``j / n_to`` of the interval, exact
        fractions of ``n_to``, so a point that is a node takes that node's
        value with weight 1 (injection, for ``n_from = 2 n_to``), and a
        midpoint the mean of its two nodes (for ``n_to = 2 n_from``).
    """
    n_from, n_to = integer("n_from", n_from, 1), integer("n_to", n_to, 1)
    left, rest = np.divmod(np.arange(n_to + 1) * n_from, n_to)
    weight = rest / n_to
    right = np.minimum(left + 1, n_from)
    rows = np.arange(n_to + 1)
    matrix = sp.csr_matrix(
        (
            np.concatenate([1.0 - weight, weight]),
            (np.concatenate([rows, rows]), np.concatenate([left, right])),
        ),
        shape=(n_to + 1, n_from + 1),
    )
    matrix.eliminate_zeros()
    return matrix


def resample(u: np.ndarray, n_to: int) -> np.ndarray:
    """Interpolate node values multilinearly onto another grid of the same box.

    Parameters
    ----------
    u
        Shape ``(n + 1,) * d``: the values at the nodes of the grid with
        ``n`` cells per side.
    n_to
        Cells per side of the other grid.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_to + 1,) * d``: :func:`interpolation_matrix` along every
        axis. Onto ``n_to = n / 2`` it is injection, from ``n = n_to / 2``
        the multilinear prolongation.
    """
    out = np.asarray(u, dtype=float)
    matrix = interpolation_matrix(out.shape[0] - 1, n_to)
    for k in range(out.ndim):
        moved = np.moveaxis(out, k, 0)
        flat = matrix @ moved.reshape(moved.shape[0], -1)
        out = np.moveaxis(flat.reshape(n_to + 1, *moved.shape[1:]), 0, k)
    return np.ascontiguousarray(out)


def half_weighting(r: np.ndarray) -> np.ndarray:
    """Return the half-weighted average of values that vanish on the boundary ring.

    At each interior node ``r/2`` plus ``r/(4 d)`` at each of its ``2 d``
    axial neighbours: in 2D ``[0 1 0; 1 4 1; 0 1 0] / 8``, in 3D the stencil
    of Liu et al. (2017), Section 3, centre ``6/12`` and faces ``1/12``. 0 on
    the ring.

    Parameters
    ----------
    r
        Shape ``(n + 1,) * d``, 0 on the boundary ring.

    Returns
    -------
    numpy.ndarray
        Same shape.
    """
    r = np.asarray(r, dtype=float)
    d = r.ndim
    inner = (slice(1, -1),) * d
    out = np.zeros_like(r)
    total = np.zeros(r[inner].shape)
    for k in range(d):
        for shift in (slice(2, None), slice(None, -2)):
            total += r[(*inner[:k], shift, *inner[k + 1 :])]
    out[inner] = 0.5 * r[inner] + total / (4.0 * d)
    return out


def cubic_resample(u: np.ndarray, n_to: int) -> np.ndarray:
    """Interpolate node values onto another grid of the same box by a tensor spline.

    The FMG interpolation of Liu et al. (2017), Algorithm 2, which says
    "cubic" and no more: the not-a-knot cubic spline through the values,
    along one axis after the other (the tensor-product spline evaluated at
    the new nodes); a grid of fewer than 3 cells per side takes degree ``n``.

    Parameters
    ----------
    u
        Shape ``(n + 1,) * d``, ``n >= 1``.
    n_to
        Cells per side of the other grid.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_to + 1,) * d``.
    """
    out = np.asarray(u, dtype=float)
    n = out.shape[0] - 1
    old, new = np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n_to + 1)
    for k in range(out.ndim):
        out = make_interp_spline(old, out, k=min(3, n), axis=k)(new)
    return np.ascontiguousarray(out)


class _MultigridScheme(Protocol):
    """The scheme methods :class:`Multigrid` calls: ``rhs`` and its ``requires``."""

    def rhs(self, grid: Any) -> np.ndarray: ...
    def values(self, grid: Any, v: np.ndarray) -> np.ndarray: ...
    def residual(self, grid: Any, v: np.ndarray) -> np.ndarray: ...
    def relax(
        self, grid: Any, u: np.ndarray, b: np.ndarray, nodes: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class Level:
    """One grid of the multigrid hierarchy.

    Attributes
    ----------
    scheme
        The scheme on this grid: the iteration's scheme with ``n`` cells per side.
    context
        ``scheme.discretize(problem)``.
    grid
        The box grid of ``n`` cells per side, the layout of the node values
        (its ``shape`` and ``interior`` only).
    f
        The right-hand side at every node, 0 on the ring.
    groups
        :func:`sweep_groups` of the grid.
    """

    scheme: Any
    context: Any
    grid: BoxGrid
    f: np.ndarray
    groups: tuple[np.ndarray, ...]


@dataclass(frozen=True, eq=False)
class MultigridContext:
    """The working context of :class:`Multigrid`: the hierarchy, finest first.

    Attributes
    ----------
    levels
        The grids from ``n`` cells per side down to 2.
    reference
        The RMS over the unknowns of the residual of the node values that
        are 0 inside, with the boundary data on the ring: the reference of
        the relative residual.
    scale
        ``max(1, max |f|)`` over the unknowns of the finest grid, the scale
        of the stopping bound.
    """

    levels: tuple[Level, ...]
    reference: float
    scale: float


@dataclass(frozen=True, kw_only=True)
class Multigrid(SchemeIteration):
    """Full multigrid with FAS V-cycles and a nonlinear Gauss-Seidel smoother.

    The method of Liu, Froese, Oberman and Xiao (2017). The grids have
    ``n``, ``ceil(n/2)``, ... cells per side down to 2, the scheme
    rediscretised on each; for ``n`` a power of 2 they are nested, as in
    the paper, and the residual goes down by its half-weighting, the
    iterate by injection and the correction comes up by multilinear
    interpolation. Other ``n`` take the same operators through
    :func:`resample`, the linear interpolation between the grids.

    The first step is one FMG cycle (Algorithm 2): the coarsest grid's one
    unknown per axis from ``coarsest_sweeps`` sweeps from 0, then on each
    finer grid the coarser solution interpolated by :meth:`interpolate_fmg`
    followed by one V-cycle. :meth:`initial_guess` does all of it below the
    finest grid, so that step 1 is the finest V-cycle, and later steps are
    further V-cycles (Algorithm 1): ``pre_sweeps`` sweeps, the FAS coarse
    problem ``S_H(w_H) = S_H(I w) + I r`` solved by the same cycle from the
    injected iterate ``w_H = I w``, its correction interpolated back, and
    ``post_sweeps`` sweeps. (Algorithm 1, step (4), also prints a
    half-weighted ``I w`` as the coarse start; from it the discrete solution
    is not a fixed point of the cycle, so ours starts from the injected one,
    ``benchmarks/multigrid.md``, Conventions.) A sweep visits the nodes in
    the ``ordering`` (:func:`sweep_groups`) and solves each node's equation
    for its own value with ``scheme.relax``, which picks the root with a
    positive definite discrete Hessian; this selects the convex solution
    where Newton's method may reach a non-convex one.

    Parameters
    ----------
    scheme
        A scheme on a box with a field ``n``, whose unknowns are the
        interior nodes of ``BoxGrid(problem.domain, n)`` in flat order and
        which provides ``values(grid, v)`` (the values at every node, the
        boundary data on the ring), ``residual(grid, v)`` and
        ``relax(grid, u, b, nodes)``, such as
        :class:`~maesolvers.classical.standard_box.StandardBox`. The coarse
        grids take ``dataclasses.replace(scheme, n=...)``.
    ordering
        ``"lexicographic"`` (the paper's 3D sweep, and its Table 2) or
        ``"red-black"`` (its Table 1).
    pre_sweeps, post_sweeps
        Sweeps before and after the coarse-grid correction, at least 0; the
        paper's 2 and 2.
    coarsest_sweeps
        Sweeps on the coarsest grid, at least 1; the paper's 1.
    max_iter
        Most cycles, the FMG cycle included.
    tol
        Stop when the RMS over the unknowns of the residual ``S(u) - f`` is
        at most ``tol * max(1, max |f|)``, a bound that does not depend on
        the grid (Newton's bound, with the RMS for its max). The history and
        the callbacks report the paper's relative residual instead, the RMS
        divided by that of the node values that are 0 inside; the paper
        stops when it is ``1e-6`` (Section 4), a rule that loosens as ``n``
        grows (``benchmarks/multigrid.md``, Finding 2).

    Raises
    ------
    TypeError
        On construction, if the scheme has no field ``n``.
    """

    measure: ClassVar[str] = "relative_residual"
    requires: ClassVar[tuple[str, ...]] = ("values", "residual", "relax")

    ordering: Ordering = "lexicographic"
    pre_sweeps: int = 2
    post_sweeps: int = 2
    coarsest_sweeps: int = 1
    max_iter: int = 50
    tol: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if "n" not in {f.name for f in dataclasses.fields(cast(Any, self.scheme))}:
            raise TypeError(
                f"{type(self).__name__} needs a scheme with a field n, the cells per "
                f"side, which {type(self.scheme).__name__} does not have"
            )
        if self.ordering not in ("lexicographic", "red-black"):
            raise ValueError(
                f"ordering must be 'lexicographic' or 'red-black', got {self.ordering!r}"
            )
        object.__setattr__(
            self, "pre_sweeps", integer("pre_sweeps", self.pre_sweeps, 0)
        )
        object.__setattr__(
            self, "post_sweeps", integer("post_sweeps", self.post_sweeps, 0)
        )
        object.__setattr__(
            self, "coarsest_sweeps", integer("coarsest_sweeps", self.coarsest_sweeps, 1)
        )

    @property
    def _scheme(self) -> _MultigridScheme:
        """The scheme, typed by the methods :meth:`check` makes sure it has."""
        return cast(_MultigridScheme, self.scheme)

    def supports(self, problem: Problem) -> bool:
        """Whether the scheme supports the problem and it is a box with equal sides.

        The pointwise smoother and the isotropic transfers need the same
        spacing along every axis: on ``Box((0, 0, 0), (1, 2, 0.5))`` a V-cycle
        cuts the residual by a factor of about 0.75 only, against 0.06 to
        0.16 on a cube (``docs/methods/multigrid.md``, Known limits).

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        domain = problem.domain
        if not isinstance(domain, Box):
            return False
        sides = np.subtract(domain.hi, domain.lo)
        return bool(np.ptp(sides) <= 1e-12 * np.max(sides)) and super().supports(
            problem
        )

    def setup(self, problem: Problem) -> MultigridContext:
        """Discretise the problem on every grid of the hierarchy.

        Parameters
        ----------
        problem
            A problem that passed :meth:`check`.

        Returns
        -------
        MultigridContext

        Raises
        ------
        ValueError
            If the scheme's unknowns are not the interior nodes of the box grid.
        """
        assert isinstance(problem.domain, Box)  # supports guarantees it
        sizes = [int(cast(Any, self.scheme).n)]  # the field __post_init__ checks
        while sizes[-1] > 2:
            sizes.append((sizes[-1] + 1) // 2)
        levels = []
        for n in sizes:
            scheme = dataclasses.replace(cast(Any, self.scheme), n=n)
            context = scheme.discretize(problem)
            grid = BoxGrid(problem.domain, n)
            rhs = cast(_MultigridScheme, scheme).rhs(context)
            if rhs.shape != (np.count_nonzero(grid.interior),):
                raise ValueError(
                    f"{type(self).__name__} needs a scheme whose unknowns are the "
                    f"interior nodes of the box grid; {type(scheme).__name__} has "
                    f"{rhs.size} at n = {n}"
                )
            f = np.zeros(math.prod(grid.shape))
            f[grid.interior] = rhs
            levels.append(
                Level(
                    scheme=scheme,
                    context=context,
                    grid=grid,
                    f=f,
                    groups=sweep_groups(n, grid.dim, self.ordering),
                )
            )
        finest = levels[0]
        reference = _norm(finest, finest.f - _operator(finest, _start(finest)))
        scale = max(1.0, float(np.max(np.abs(finest.f))))
        return MultigridContext(levels=tuple(levels), reference=reference, scale=scale)

    def initial_guess(self, problem: Problem, context: MultigridContext) -> np.ndarray:
        """FMG below the finest grid, interpolated onto it (Algorithm 2 but its last V-cycle).

        On a grid of 2 cells per side, the hierarchy's only grid, the values
        0 inside instead.

        Parameters
        ----------
        problem
            The problem.
        context
            From :meth:`setup`.

        Returns
        -------
        numpy.ndarray
            The unknowns.
        """
        levels = context.levels
        if len(levels) == 1:
            return _start(levels[0])[levels[0].grid.interior]
        u = self._interpolate(levels[0], levels[1], self._full_multigrid(levels, 1))
        return u[levels[0].grid.interior]

    def step(
        self, problem: Problem, context: MultigridContext, iterate: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """One V-cycle on the finest grid.

        Parameters
        ----------
        problem
            The problem.
        context
            From :meth:`setup`.
        iterate
            The current unknowns.

        Returns
        -------
        tuple or None
            The next unknowns and their relative residual; None if the
            residual is not finite.
        """
        finest = context.levels[0]
        u = self._scheme.values(finest.context, iterate)
        u = self._cycle(context.levels, 0, u, finest.f)
        measure = _norm(finest, finest.f - _operator(finest, u)) / context.reference
        if not np.isfinite(measure):
            return None
        return u[finest.grid.interior], measure

    def converged(
        self, problem: Problem, context: MultigridContext, measure: float
    ) -> bool:
        """Whether the residual's RMS is at most ``tol * max(1, max |f|)``.

        Parameters
        ----------
        problem
            The problem.
        context
            From :meth:`setup`.
        measure
            The relative residual: the residual's RMS over the context's
            ``reference``.

        Returns
        -------
        bool
        """
        return measure * context.reference <= self.tol * context.scale

    def finalize(
        self, problem: Problem, context: MultigridContext, state: np.ndarray
    ) -> Solution:
        """Return the scheme's solution fields on the finest grid.

        Parameters
        ----------
        problem
            The problem.
        context
            From :meth:`setup`.
        state
            The final unknowns.

        Returns
        -------
        Solution
        """
        finest = context.levels[0]
        return self.scheme.solution(finest.context, state)

    def interpolate_fmg(self, u: np.ndarray, n_to: int) -> np.ndarray:
        """Interpolate a coarse grid's solution onto the next finer grid in FMG.

        The paper's "cubic interpolation" (Algorithm 2), which it does not
        specify further: :func:`cubic_resample`, the tensor-product
        not-a-knot spline. A subclass may override it; the boundary ring of
        the result is replaced by the boundary data.

        Parameters
        ----------
        u
            Shape ``(n + 1,) * d``: the node values of the coarse grid.
        n_to
            Cells per side of the finer grid.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_to + 1,) * d``.
        """
        return cubic_resample(u, n_to)

    def _full_multigrid(self, levels: Sequence[Level], index: int) -> np.ndarray:
        """Algorithm 2 on ``levels[index]``: the FMG solution there."""
        level = levels[index]
        if index == len(levels) - 1:
            u = _start(level)
            _sweep(level, u, level.f, self.coarsest_sweeps)
            return u
        coarse = levels[index + 1]
        u = self._interpolate(level, coarse, self._full_multigrid(levels, index + 1))
        return self._cycle(levels, index, u, level.f)

    def _interpolate(
        self, fine: Level, coarse: Level, u_coarse: np.ndarray
    ) -> np.ndarray:
        """:meth:`interpolate_fmg` onto ``fine``, with the boundary data on its ring."""
        u = self.interpolate_fmg(u_coarse.reshape(coarse.grid.shape), fine.grid.n)
        return _with_ring(fine, u.ravel())

    def _cycle(
        self, levels: Sequence[Level], index: int, u: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Algorithm 1 on ``levels[index]`` from ``u`` (changed in place) for ``S(u) = b``."""
        level = levels[index]
        if index == len(levels) - 1:
            _sweep(level, u, b, self.coarsest_sweeps)
            return u
        _sweep(level, u, b, self.pre_sweeps)
        coarse = levels[index + 1]
        shape, n_coarse = level.grid.shape, coarse.grid.n
        residual = resample(
            half_weighting((b - _operator(level, u)).reshape(shape)), n_coarse
        )
        residual = residual.ravel()
        residual[~coarse.grid.interior] = 0.0
        w = _with_ring(coarse, resample(u.reshape(shape), n_coarse).ravel())
        corrected = self._cycle(
            levels, index + 1, w.copy(), _operator(coarse, w) + residual
        )
        delta = resample(
            (corrected - w).reshape(coarse.grid.shape), level.grid.n
        ).ravel()
        u += delta
        _sweep(level, u, b, self.post_sweeps)
        return u


# The cycles work on the values at every node of a level, in flat order;
# a right-hand side b is 0 on the ring.


def _with_ring(level: Level, u: np.ndarray) -> np.ndarray:
    """Return node values with the scheme's boundary data on the ring."""
    scheme = cast(_MultigridScheme, level.scheme)
    return scheme.values(level.context, u[level.grid.interior])


def _start(level: Level) -> np.ndarray:
    """Return the node values 0 inside, with the boundary data on the ring."""
    scheme = cast(_MultigridScheme, level.scheme)
    return scheme.values(level.context, np.zeros(np.count_nonzero(level.grid.interior)))


def _operator(level: Level, u: np.ndarray) -> np.ndarray:
    """Return ``S(u)`` at the interior nodes, 0 on the ring."""
    out = np.zeros_like(u)
    scheme = cast(_MultigridScheme, level.scheme)
    v = u[level.grid.interior]
    out[level.grid.interior] = scheme.residual(level.context, v) + scheme.rhs(
        level.context
    )
    return out


def _sweep(level: Level, u: np.ndarray, b: np.ndarray, sweeps: int) -> None:
    """Run Gauss-Seidel sweeps for ``S(u) = b``, changing ``u`` in place."""
    scheme = cast(_MultigridScheme, level.scheme)
    for _ in range(sweeps):
        for nodes in level.groups:
            u[nodes] = scheme.relax(level.context, u, b[nodes], nodes)


def _norm(level: Level, r: np.ndarray) -> float:
    """Return the RMS of node values ``r`` over the unknowns (the discrete L2 norm, scaled)."""
    return float(np.sqrt(np.mean(np.square(r[level.grid.interior]))))
