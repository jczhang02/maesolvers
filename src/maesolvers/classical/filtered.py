"""The filtered scheme of Froese and Oberman (2013).

A filtered scheme (Froese-Oberman, SIAM J. Numer. Anal. 51 (2013), Section
4.1) combines an accurate scheme ``F_A`` with a monotone one ``F_M``:

    F_H[u] = F_M[u] + eps S((F_A[u] - F_M[u]) / eps),

with the filter ``S`` of their eq. (1.3) (:func:`filter_function`): the
identity on ``[-1, 1]``, zero outside ``[-2, 2]``, linear between. Where the
two schemes nearly agree, as on a smooth solution, ``F_H = F_A``; where they
differ by more than ``2 eps``, near a singularity, ``F_H = F_M``. As
``|F_H - F_M| <= eps``, the scheme is nearly monotone, and its solutions
converge to the viscosity solution as ``eps`` goes to zero (their Theorem 1).

The kernels :func:`filter_function`, :func:`filter_slope` and
:func:`filter_width` are module-level pure functions. :class:`Filtered`
composes any accurate and any monotone
:class:`~maesolvers.classical.schemes.Scheme` that share their unknowns into
a scheme that :class:`~maesolvers.classical.iterations.Newton` solves.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from maesolvers.classical.schemes import Scheme
from maesolvers.core._params import positive
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError, check_part

_ACCURATE_NEEDS = ("residual", "jacobian")
_MONOTONE_NEEDS = ("residual", "jacobian", "solve_laplace")


def filter_function(x: np.ndarray) -> np.ndarray:
    """Return the filter ``S`` of Froese-Oberman (2013), eq. (1.3).

    ``S(x) = x`` for ``|x| <= 1``, ``sign(x) (2 - |x|)`` for
    ``1 <= |x| <= 2``, and 0 for ``|x| >= 2``: continuous, odd, bounded by 1.

    Parameters
    ----------
    x
        Any shape.

    Returns
    -------
    numpy.ndarray
        Same shape.
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.clip(np.minimum(np.abs(x), 2.0 - np.abs(x)), 0.0, None)


def filter_slope(x: np.ndarray) -> np.ndarray:
    """Return the derivative ``S'`` of :func:`filter_function`.

    1 for ``|x| <= 1``, -1 for ``1 < |x| < 2`` and 0 for ``|x| >= 2``: at
    ``|x| = 1`` it takes 1, the accurate side, and at ``|x| = 2`` it takes 0,
    the monotone side.

    Parameters
    ----------
    x
        Any shape.

    Returns
    -------
    numpy.ndarray
        Same shape, values in ``{-1, 0, 1}``.
    """
    a = np.abs(np.asarray(x, dtype=float))
    return np.where(a <= 1.0, 1.0, np.where(a < 2.0, -1.0, 0.0))


def filter_width(h: float, dtheta: float) -> float:
    """Return Froese and Oberman's filter width ``eps = sqrt(h) + dtheta / 10``.

    Their choice for the computations of Section 4 (Froese-Oberman 2013,
    p. 439). It goes to zero with ``h`` and ``dtheta``, more slowly than the
    accurate scheme's error, so the filter keeps the accurate scheme on a
    smooth solution (their Lemma 4).

    Parameters
    ----------
    h
        Grid spacing, positive.
    dtheta
        Angular resolution of the monotone scheme's stencil, nonnegative,
        in radians. The paper gives it no value; :class:`Filtered` passes
        the largest angle between neighbouring stencil directions.

    Returns
    -------
    float
    """
    return float(np.sqrt(h) + dtheta / 10.0)


@dataclass(frozen=True, eq=False)
class FilteredGrid:
    """One problem discretised by :class:`Filtered`, its working context.

    Only :class:`Filtered` reads the fields; they are not a stable interface.

    Attributes
    ----------
    accurate
        The accurate scheme's grid.
    monotone
        The monotone scheme's grid.
    width
        The filter width ``eps``.
    """

    accurate: Any
    monotone: Any
    width: float


@dataclass(frozen=True, kw_only=True)
class Filtered(Scheme):
    """The filtered combination of an accurate and a monotone scheme.

    At every unknown the discrete equation is

        F_M[u] + eps S((F_A[u] - F_M[u]) / eps) = 0,

    with ``F_A`` and ``F_M`` the residuals of the two parts and ``S``
    :func:`filter_function` (Froese-Oberman 2013, Section 4.1). Where
    ``|F_A - F_M| <= eps`` it is the accurate scheme's equation, where
    ``|F_A - F_M| >= 2 eps`` the monotone scheme's.

    The two parts must discretise a problem on the same unknowns, in the same
    order: :meth:`discretize` checks that their right-hand sides agree.
    ``Filtered(accurate=Standard9(n=n), monotone=Monotone17(n=n, ...))`` is
    the scheme of the paper.

    Parameters
    ----------
    accurate
        A scheme with ``residual(grid, v)`` and ``jacobian(grid, v)``,
        typically consistent to higher order but not monotone, such as
        :class:`~maesolvers.classical.schemes.Standard9`. It also gives the
        solution fields (nodes, the discrete Hessian, the interpolant).
    monotone
        A scheme with ``residual``, ``jacobian`` and
        ``solve_laplace(grid, rhs)``, such as
        :class:`~maesolvers.classical.monotone.Monotone17`; Newton starts
        from its Laplacian.
    epsilon
        The filter width ``eps``, positive; None (the default) takes
        Froese and Oberman's ``sqrt(h) + dtheta / 10`` (:func:`filter_width`),
        with ``h`` the bounding box's longest side over ``monotone.n`` and
        ``dtheta = 2 * monotone.angular_resolution``, the largest angle
        between neighbouring stencil directions (``pi/4``, ``atan(1/2)``,
        ``atan(1/3)`` for 9, 17 and 33 points), so the monotone part must
        then have both attributes. The paper leaves ``dtheta`` undefined;
        its T2 columns of Table 4.1 fit this full gap, not Oberman's (2008)
        half gap ``angular_resolution`` (``benchmarks/filtered.md``,
        Finding 3).

    Raises
    ------
    TypeError
        On construction, if a part is not an instance of a frozen-dataclass
        :class:`~maesolvers.classical.schemes.Scheme` subclass, lacks a
        method named above, or, with ``epsilon=None``, the monotone part has
        no ``n`` or ``angular_resolution``.
    ValueError
        On construction, if ``epsilon`` is not positive, or both parts have
        an ``n`` and they differ.
    """

    accurate: Scheme
    monotone: Scheme
    epsilon: float | None = None

    def __post_init__(self) -> None:
        for name, needs in (
            ("accurate", _ACCURATE_NEEDS),
            ("monotone", _MONOTONE_NEEDS),
        ):
            part = getattr(self, name)
            check_part(f"Filtered.{name}", part, Scheme)
            missing = [m for m in needs if not callable(getattr(part, m, None))]
            if missing:
                raise TypeError(
                    f"Filtered.{name} needs {', '.join(missing)}, which "
                    f"{type(part).__name__} does not have"
                )
        if self.epsilon is None:
            missing = [
                a for a in ("n", "angular_resolution") if not hasattr(self.monotone, a)
            ]
            if missing:
                raise TypeError(
                    f"Filtered needs an epsilon, or a monotone scheme with n and "
                    f"angular_resolution; {type(self.monotone).__name__} has no "
                    f"{' or '.join(missing)}"
                )
        else:
            object.__setattr__(self, "epsilon", positive("epsilon", self.epsilon))
        n_accurate = getattr(self.accurate, "n", None)
        n_monotone = getattr(self.monotone, "n", None)
        if None not in (n_accurate, n_monotone) and n_accurate != n_monotone:
            raise ValueError(
                f"Filtered needs its parts on one grid, got n = {n_accurate} "
                f"({type(self.accurate).__name__}) and n = {n_monotone} "
                f"({type(self.monotone).__name__})"
            )

    def supports(self, problem: Problem) -> bool:
        """Whether both parts support the problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return self.accurate.supports(problem) and self.monotone.supports(problem)

    def discretize(self, problem: Problem) -> FilteredGrid:
        """Discretise the problem with both parts and fix the filter width.

        Parameters
        ----------
        problem
            A problem both parts support.

        Returns
        -------
        FilteredGrid

        Raises
        ------
        UnsupportedProblemError
            If a part does not support the problem.
        ValueError
            If the parts' right-hand sides differ (beyond a relative ``1e-12``),
            so they do not share their unknowns.
        """
        for name in ("accurate", "monotone"):
            part = getattr(self, name)
            if not part.supports(problem):
                raise UnsupportedProblemError(
                    f"Filtered does not support {problem.name}: its {name} "
                    f"scheme {type(part).__name__} does not"
                )
        accurate = self.accurate.discretize(problem)
        monotone = self.monotone.discretize(problem)
        fa, fm = self.accurate.rhs(accurate), self.monotone.rhs(monotone)
        if fa.shape != fm.shape or not np.allclose(fa, fm, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"Filtered needs its parts on the same unknowns, but on "
                f"{problem.name} {type(self.accurate).__name__} has {fa.size} and "
                f"{type(self.monotone).__name__} {fm.size}, or their f differ"
            )
        if self.epsilon is None:
            part: Any = self.monotone
            side = float(
                np.max(np.subtract(problem.domain.bbox.hi, problem.domain.bbox.lo))
            )
            width = filter_width(side / part.n, 2.0 * part.angular_resolution)
        else:
            width = self.epsilon
        return FilteredGrid(accurate=accurate, monotone=monotone, width=width)

    def rhs(self, grid: FilteredGrid) -> np.ndarray:
        """Return the right-hand side ``f`` at the unknowns.

        Parameters
        ----------
        grid
            From :meth:`discretize`.

        Returns
        -------
        numpy.ndarray
        """
        return self.monotone.rhs(grid.monotone)

    def _residuals(self, grid: FilteredGrid, v: np.ndarray) -> tuple[np.ndarray, ...]:
        """``F_A``, ``F_M`` and the filter's argument ``(F_A - F_M) / eps``."""
        accurate: Any = self.accurate
        monotone: Any = self.monotone
        fa = accurate.residual(grid.accurate, v)
        fm = monotone.residual(grid.monotone, v)
        return fa, fm, (fa - fm) / grid.width

    def residual(self, grid: FilteredGrid, v: np.ndarray) -> np.ndarray:
        """``F_M + eps S((F_A - F_M) / eps)`` at the unknowns.

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
        _, fm, x = self._residuals(grid, v)
        return fm + grid.width * filter_function(x)

    def jacobian(self, grid: FilteredGrid, v: np.ndarray) -> sp.csc_matrix:
        """Froese and Oberman's approximate Jacobian of :meth:`residual`.

        ``diag(1 - S') J_M + diag(max(S', 0)) J_A`` (Froese-Oberman 2013,
        p. 438): the exact derivative where ``S' = 1`` (the accurate
        scheme's Jacobian) or ``S' = 0`` (the monotone scheme's), and
        ``2 J_M`` instead of ``2 J_M - J_A`` in the band
        ``1 < |F_A - F_M| / eps < 2``, which they choose because a negative
        weight on ``J_A`` can make the linear systems ill-posed.

        It evaluates both parts' residuals again, which :meth:`residual` has
        done at the same iterate: one more residual evaluation per Newton
        step, the price of a scheme that holds no state between calls.

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
        accurate: Any = self.accurate
        monotone: Any = self.monotone
        slope = filter_slope(self._residuals(grid, v)[2])
        ja = accurate.jacobian(grid.accurate, v)
        jm = monotone.jacobian(grid.monotone, v)
        return sp.csc_matrix(
            sp.diags(1.0 - slope) @ jm + sp.diags(np.maximum(slope, 0.0)) @ ja
        )

    def solve_laplace(self, grid: FilteredGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplace u = rhs`` with the Dirichlet data, by the monotone part.

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
        monotone: Any = self.monotone
        return monotone.solve_laplace(grid.monotone, rhs)

    def solution(self, grid: FilteredGrid, v: np.ndarray) -> Solution:
        """Return the accurate part's solution fields, with the filtered residual.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        Solution
            The accurate scheme's solution, whose ``info`` gets
            ``residual_max``, the max absolute value of :meth:`residual`;
            ``filter_width``, ``eps``; and ``monotone_fraction``, the share
            of unknowns where ``|F_A - F_M| > eps``, so that the filter
            departs from the accurate scheme (Froese-Oberman 2013, Fig. 4.3).
        """
        solution = self.accurate.solution(grid.accurate, v)
        _, fm, x = self._residuals(grid, v)
        residual = fm + grid.width * filter_function(x)
        info = {
            **solution.info,
            "residual_max": float(np.max(np.abs(residual))),
            "filter_width": grid.width,
            "monotone_fraction": float(np.mean(np.abs(x) > 1.0)),
        }
        return dataclasses.replace(solution, info=info)
