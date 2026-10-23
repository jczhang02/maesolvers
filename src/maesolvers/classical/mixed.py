"""The monotone mixed scheme of Chen, Wan and Lin (2018).

The scheme (Chen-Wan-Lin, J. Sci. Comput. 76 (2018) 1839-1867) solves the
Monge-Ampere equation of a convex function in its HJB form (their Theorem 1
and eq. (7)): with the control ``(a, theta)`` in ``[0, 1] x [-pi/4, pi/4)``,

    max over (a, theta) of
        -a11 u_xx - 2 a12 u_xy - a22 u_yy + 2 sqrt(a (1 - a) f) = 0,

    a11 = (1 - (1 - 2a) cos 2 theta) / 2,
    a22 = (1 + (1 - 2a) cos 2 theta) / 2,
    a12 = (1 - 2a) sin 2 theta / 2,

the coefficients of the trace-one matrix with eigenvalues ``a`` and
``1 - a`` along the axes turned by ``-theta`` (their eq. (6), (8)). For a
fixed control the operator is linear, and the maximum is 0 exactly at a
convex solution, so no convexity constraint is needed.

The discretisation depends on the control. Where ``a11 >= |a12|`` and
``a22 >= |a12|`` the standard 7-point stencil, second order, is monotone
(with the diagonal ``(1, 1)`` where ``a12 >= 0``, ``(1, -1)`` where
``a12 <= 0``; their eq. (12) to (15)); for the other controls the
semi-Lagrangian wide stencil, first order, is: the second differences along
the turned axes with arms of length ``sqrt(h)``, the values at the arms'
ends interpolated bilinearly, an arm that leaves the box shortened to it
with the Dirichlet data there (their eq. (16) to (19)). At every node the
discrete equation is the maximum over the controls, each with its own
stencil, so every linearisation is an M-matrix, and the paper proves
convergence to the viscosity solution (their Section 6).

The maximum over the 7-point controls has closed forms (their eq. (30) to
(34)); over the wide-stencil controls it is a search over ``M`` angles, each
with its best ``a`` in closed form (their eq. (36) to (38)). Where the
stationary point of a 7-point objective, eq. (30), lies in its region, it
maximises that objective over all controls, and the paper's algorithm takes
it without the wide-stencil search; its tables match only this rule
(``benchmarks/chen_wan_lin2018.md``, Finding 2).
:class:`MixedHJB` composes these into a
:class:`~maesolvers.classical.schemes.Scheme` that
:class:`~maesolvers.classical.iterations.PolicyIteration` (Howard's
algorithm, the paper's) solves.

The numerical kernels are module-level pure functions:
:func:`control_coefficients`, :func:`seven_point_case`,
:func:`seven_point_band`, :func:`seven_point_controls`,
:func:`wide_stencil_weight` and :func:`wide_stencil_value`.
"""

# No `from __future__ import annotations` here: jsonargparse reads the field
# types (ADR 0005); see maesolvers.classical.iterations.

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from maesolvers.classical._grid import interior_mask, spline
from maesolvers.classical.schemes import Scheme, difference_operators
from maesolvers.core._params import integer
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box, Field
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, UnsupportedProblemError


def control_coefficients(
    a: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the coefficients ``a11``, ``a22``, ``a12`` of the control ``(a, theta)``.

    Chen-Wan-Lin (2018), eq. (8): the entries of the trace-one matrix
    ``R diag(a, 1 - a) R^T`` with ``R`` the rotation by ``-theta``.

    Parameters
    ----------
    a, theta
        Arrays of one shape; ``a`` in ``[0, 1]``.

    Returns
    -------
    tuple of numpy.ndarray
        ``a11``, ``a22``, ``a12``, of the shape of ``a`` and ``theta``.
    """
    b = 1.0 - 2.0 * np.asarray(a, dtype=float)
    c, s = np.cos(2.0 * np.asarray(theta)), np.sin(2.0 * np.asarray(theta))
    return 0.5 * (1.0 - b * c), 0.5 * (1.0 + b * c), 0.5 * b * s


def seven_point_case(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return which 7-point stencil is monotone for each control, if any.

    Chen-Wan-Lin (2018), conditions (12) and (14): ``a11 >= |a12|`` and
    ``a22 >= |a12|``, with ``a12 >= 0`` for the stencil with the diagonal
    ``(1, 1)`` (case 1) and ``a12 <= 0`` for the one with ``(1, -1)``
    (case 2).

    Parameters
    ----------
    a, theta
        Arrays of one shape; ``a`` in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Of int, the shape of ``a``: 1 for case 1 (also where ``a12 = 0``,
        where both cases are one 5-point stencil), 2 for case 2, 0 where
        neither stencil is monotone and the wide stencil is used.
    """
    a11, a22, a12 = control_coefficients(a, theta)
    dominant = (a11 >= np.abs(a12)) & (a22 >= np.abs(a12))
    return np.where(dominant, np.where(a12 >= 0.0, 1, 2), 0)


def seven_point_band(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the band of ``a`` in which a 7-point stencil is monotone at angle ``theta``.

    Chen-Wan-Lin (2018), eq. (37): ``C^-`` and ``C^+``,
    ``(1 -/+ 1 / (sqrt 2 sin(2 |theta| + pi/4))) / 2``. A control
    ``(a, theta)`` meets condition (12) or (14) exactly when
    ``C^- <= a <= C^+``; the band is ``[0, 1]`` at ``theta = 0`` and
    ``theta = -/+ pi/4``, where the turned axes are the grid's axes or
    diagonals, and narrowest, ``[0.146, 0.854]``, at ``theta = -/+ pi/8``.

    Parameters
    ----------
    theta
        Angles in ``[-pi/4, pi/4]``.

    Returns
    -------
    tuple of numpy.ndarray
        ``C^-`` and ``C^+``, of the shape of ``theta``.
    """
    half_width = 0.5 / (np.sqrt(2.0) * np.sin(2.0 * np.abs(theta) + 0.25 * np.pi))
    return 0.5 - half_width, 0.5 + half_width


def _hjb_term(a: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``2 sqrt(a (1 - a) f)``, the control's share of the right-hand side."""
    return 2.0 * np.sqrt(np.maximum(a * (1.0 - a), 0.0) * f)


def _seven_point_value(
    a: np.ndarray,
    theta: np.ndarray,
    dxx: np.ndarray,
    dyy: np.ndarray,
    dxy: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """Evaluate the objective of eq. (29) at the control ``(a, theta)``."""
    a11, a22, a12 = control_coefficients(a, theta)
    return -a11 * dxx - 2.0 * a12 * dxy - a22 * dyy + _hjb_term(a, f)


def _best_weight(lam: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``(1 - lam / sqrt(4 f + lam^2)) / 2``, the ``a`` that maximises for given ``lam``.

    Where ``f = 0`` and ``lam = 0`` every ``a`` is optimal; it returns 1/2.
    """
    root = np.sqrt(4.0 * f + lam * lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(root > 0.0, 0.5 * (1.0 - lam / root), 0.5)


def _boundary_angle(gamma: np.ndarray) -> np.ndarray:
    """``arctan(1 + gamma^2 - gamma sqrt(2 + gamma^2))``, eq. (33), without cancellation.

    For ``gamma >= 0`` the argument is ``1 / (1 + gamma^2 + gamma sqrt(2 +
    gamma^2))``, the same number; ``gamma = +inf`` gives 0, ``-inf`` gives
    ``pi/2``, and NaN (``f = 0`` with a zero numerator) is taken as 0.
    """
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=np.inf, neginf=-np.inf)
    with np.errstate(invalid="ignore", over="ignore"):
        root = np.sqrt(2.0 + gamma * gamma)
        positive = 1.0 / (1.0 + gamma * gamma + gamma * root)
        negative = 1.0 + gamma * gamma - gamma * root
    return np.arctan(np.where(gamma >= 0.0, positive, negative))


class SevenPointControls(NamedTuple):
    """What :func:`seven_point_controls` returns; it unpacks as a tuple.

    Attributes
    ----------
    value : numpy.ndarray
        The largest value of the objective over the 7-point controls.
    a, theta : numpy.ndarray
        The control that attains it.
    case : numpy.ndarray
        Of int: the stencil, 1 or 2 as in :func:`seven_point_case`.
    stationary : numpy.ndarray
        Boolean: where the stationary point of eq. (30) of a stencil lies in
        that stencil's region. It maximises that stencil's objective over all
        controls, so the paper takes it without a wide-stencil search there.
    """

    value: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    case: np.ndarray
    stationary: np.ndarray


def seven_point_controls(
    dxx: np.ndarray,
    dyy: np.ndarray,
    dxy1: np.ndarray,
    dxy2: np.ndarray,
    f: np.ndarray,
) -> SevenPointControls:
    """Maximise the objective over the controls for which a 7-point stencil is monotone.

    Chen-Wan-Lin (2018), Section 5.2: the control set splits into the
    regions where condition (12) holds (stencil 1), where (14) holds
    (stencil 2), and the rest. The largest value over the first two is the
    largest of seven candidates, each a closed form: the stationary point of
    each region, eq. (30), kept only if it lies in its region; the line
    ``theta = 0``, eq. (31); and the two sections of each region's boundary
    with the rest, eq. (33) and (34).

    Parameters
    ----------
    dxx, dyy
        The centred second differences ``u_xx``, ``u_yy`` (their eq. (11)).
    dxy1, dxy2
        The 7-point cross differences of stencils 1 and 2 (their eq. (13),
        (15)).
    f
        The right-hand side, at least 0.

    Returns
    -------
    SevenPointControls
        The value, the control and where a stationary point is in its
        region, per point, of the shape of ``dxx``.
    """
    dxy = {1: dxy1, 2: dxy2}
    candidates = []  # (a, theta, case)
    stationary = np.zeros(np.shape(dxx), dtype=bool)
    infeasible = []  # per candidate: where it lies outside its region
    for case in (1, 2):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = 2.0 * dxy[case] / (dyy - dxx)
        theta = 0.5 * np.arctan(np.nan_to_num(ratio, nan=0.0))
        lam = (dxx - dyy) * np.cos(2.0 * theta) - 2.0 * dxy[case] * np.sin(2.0 * theta)
        a = _best_weight(lam, f)
        inside = seven_point_case(a, theta) == case
        stationary |= inside
        candidates.append((a, theta, case))
        infeasible.append(~inside)
    candidates.append((_best_weight(dxx - dyy, f), np.zeros_like(dxx), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        root_f = 2.0 * np.sqrt(f)
        for case, signs in ((1, ((1, -1), (-1, 1))), (2, ((1, 1), (-1, -1)))):
            for s_a, s_theta in signs:
                gamma = s_a * (dyy - dxx - 2.0 * s_theta * dxy[case]) / root_f
                theta = 0.5 * s_theta * _boundary_angle(gamma)
                low, high = seven_point_band(theta)
                candidates.append((high if s_a > 0 else low, theta, case))
    infeasible += [np.zeros(np.shape(dxx), dtype=bool)] * (len(candidates) - 2)
    values = np.stack(
        [
            np.where(out, -np.inf, _seven_point_value(a, t, dxx, dyy, dxy[c], f))
            for (a, t, c), out in zip(candidates, infeasible, strict=True)
        ]
    )
    best = np.argmax(values, axis=0)[None]

    def pick(field: int) -> np.ndarray:
        stacked = np.stack(
            [np.broadcast_to(c[field], np.shape(dxx)) for c in candidates]
        )
        return np.take_along_axis(stacked, best, axis=0)[0]

    return SevenPointControls(
        np.take_along_axis(values, best, axis=0)[0],
        pick(0),
        pick(1),
        pick(2).astype(int),
        stationary,
    )


def wide_stencil_weight(
    dzz: np.ndarray,
    dww: np.ndarray,
    f: np.ndarray,
    theta: np.ndarray,
    restricted: bool = True,
) -> np.ndarray:
    """Return the best ``a`` for the wide stencil at angle ``theta``.

    Chen-Wan-Lin (2018), eq. (36) to (38): the objective
    ``-a dzz - (1 - a) dww + 2 sqrt(a (1 - a) f)`` is concave in ``a`` and
    largest at ``C^lambda = (1 - (dzz - dww) / sqrt(4 f + (dzz - dww)^2)) / 2``.
    Restricted to the wide-stencil controls, ``a`` outside the band
    ``(C^-, C^+)`` of :func:`seven_point_band`, the maximiser is
    ``C^lambda`` if it lies outside the band, and otherwise the band's edge
    on the side of ``1/2`` where ``C^lambda`` lies.

    Parameters
    ----------
    dzz, dww
        The wide-stencil second differences along the turned axes ``e_z``
        and ``e_w`` (their eq. (17), (18)).
    f
        The right-hand side, at least 0.
    theta
        The angle of the control, broadcast against ``dzz``.
    restricted
        If False, ``a`` ranges over ``[0, 1]`` and the result is
        ``C^lambda``: the pure semi-Lagrangian scheme, wide at every control.

    Returns
    -------
    numpy.ndarray
        ``a`` in ``[0, 1]``, of the shape of ``dzz``.
    """
    a = _best_weight(dzz - dww, f)
    if not restricted:
        return a
    low, high = seven_point_band(np.asarray(theta, dtype=float))
    outside = (a <= low) | (a >= high)
    return np.where(outside, a, np.where(a <= 0.5, low, high))


def wide_stencil_value(
    dzz: np.ndarray, dww: np.ndarray, a: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Return the objective ``-a dzz - (1 - a) dww + 2 sqrt(a (1 - a) f)``, eq. (35).

    Parameters
    ----------
    dzz, dww
        The wide-stencil second differences along ``e_z`` and ``e_w``.
    a
        The weight of ``e_z``, in ``[0, 1]``.
    f
        The right-hand side, at least 0.

    Returns
    -------
    numpy.ndarray
    """
    return -a * dzz - (1.0 - a) * dww + _hjb_term(a, f)


def _cross_differences(n: int, h: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build the 7-point cross differences of eq. (13) and (15), all nodes to interior nodes.

    Same layout as :func:`~maesolvers.classical.schemes.difference_operators`.
    """
    m = n + 1
    index = np.flatnonzero(interior_mask(n).ravel())
    rows = np.arange(index.size)
    axes = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def build(sign: float, diagonals: list[tuple[int, int]]) -> sp.csr_matrix:
        offsets = [((0, 0), 2.0 * sign)]
        offsets += [(d, -sign) for d in axes] + [(d, sign) for d in diagonals]
        r = np.tile(rows, len(offsets))
        c = np.concatenate([index + di * m + dj for (di, dj), _ in offsets])
        v = np.concatenate([np.full(rows.size, w) for _, w in offsets])
        return sp.csr_matrix((v / (2.0 * h * h), (r, c)), shape=(index.size, m * m))

    return build(1.0, [(1, 1), (-1, -1)]), build(-1.0, [(1, -1), (-1, 1)])


@dataclass(frozen=True, eq=False)
class MixedGrid:
    """One problem discretised by :class:`MixedHJB`, its working context.

    Only :class:`MixedHJB` reads the fields; they are not a stable interface
    and may change with the scheme. The unknowns are the values at the
    ``m = (n - 1)^2`` interior nodes; the boundary nodes carry the Dirichlet
    data.

    Attributes
    ----------
    n
        Cells per side.
    h
        The grid step.
    x, y
        Node coordinates along each axis, ``n + 1`` each.
    nodes
        Shape ``((n + 1)^2, 2)``, node ``(i, j)`` at flat index
        ``i * (n + 1) + j``.
    interior
        Shape ``((n + 1)^2,)``, boolean: the nodes that carry unknowns.
    index
        Shape ``(m, 2)``, int: the grid indices ``(i, j)`` of the unknowns.
    f
        The right-hand side at the unknowns.
    g
        The Dirichlet data, for the ends of the wide stencil's arms on the
        boundary.
    boundary_values
        The Dirichlet data at the boundary nodes, in flat order.
    dxx, dyy, dxy1, dxy2
        The 7-point operators from the values at all nodes to the
        differences at the unknowns, shape ``(m, (n + 1)^2)``.
    arm
        The length ``sqrt(h)`` of the wide stencil's arms.
    angles
        The ``M`` angles of the wide-stencil search, in ``[-pi/4, pi/4)``.
    wide_search
        :attr:`MixedHJB.wide_search`.
    laplace
        Solves the 5-point ``(D_xx + D_yy) v = rhs`` by a sparse LU
        factorisation.
    """

    n: int
    h: float
    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    interior: np.ndarray
    index: np.ndarray
    f: np.ndarray
    g: Field
    boundary_values: np.ndarray
    dxx: sp.csr_matrix
    dyy: sp.csr_matrix
    dxy1: sp.csr_matrix
    dxy2: sp.csr_matrix
    arm: float
    angles: np.ndarray
    wide_search: str
    laplace: Callable[[np.ndarray], np.ndarray]


class _Arm(NamedTuple):
    """One arm of the wide stencil at a set of unknowns.

    ``length`` is ``sqrt(h)``, or less where the arm was shortened to the
    boundary. The value at the arm's end is ``sum(weights * u[cell]) +
    data``: bilinear interpolation of the node values ``u`` (flat order)
    over the cell ``cell`` where the end lies in the box, and ``data``, the
    Dirichlet data, where it was shortened (``weights`` 0 there).
    """

    length: np.ndarray
    cell: np.ndarray
    weights: np.ndarray
    data: np.ndarray


def _arm(grid: MixedGrid, index: np.ndarray, direction: np.ndarray) -> _Arm:
    """Build the arms from the nodes ``index`` (shape ``(k, 2)``) along the unit ``direction``."""
    n = grid.n
    step = np.broadcast_to(grid.arm / grid.h * direction, index.shape)
    reach = np.ones(len(index))
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis in range(2):
            d = step[:, axis]
            room = np.where(d > 0.0, (n - index[:, axis]) / d, index[:, axis] / -d)
            reach = np.minimum(reach, np.where(d != 0.0, room, np.inf))
    end = index + reach[:, None] * step
    shortened = reach < 1.0
    base = np.clip(np.floor(end), 0, n - 1).astype(int)
    fx, fy = (end - base).T
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    corner = base[:, 0] * (n + 1) + base[:, 1]
    cell = corner[:, None] + np.array([0, n + 1, 1, n + 2])
    weights[shortened] = 0.0
    data = np.zeros(len(index))
    if np.any(shortened):
        points = np.stack([grid.x[0], grid.y[0]]) + end[shortened] * grid.h
        data[shortened] = grid.g(points)
    return _Arm(reach * grid.arm, cell, weights, data)


def _arm_value(arm: _Arm, u: np.ndarray) -> np.ndarray:
    """Return the value at the end of each arm, given the node values ``u`` in flat order."""
    return np.sum(arm.weights * u[arm.cell], axis=1) + arm.data


def _directions(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``e_z`` and ``e_w``, the axes turned by ``-theta``, each of shape ``(..., 2)``."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)


def _arm_weights(plus: _Arm, minus: _Arm) -> tuple[np.ndarray, np.ndarray]:
    """Return the weights of the arms' ends in the second difference, eq. (17) and (19).

    ``D = w+ (u+ - u0) + w- (u- - u0)`` with ``w+- = 2 / (l+- (l+ + l-))``.
    """
    total = plus.length + minus.length
    return 2.0 / (plus.length * total), 2.0 / (minus.length * total)


def _wide_differences(
    grid: MixedGrid, index: np.ndarray, theta: np.ndarray, u: np.ndarray, u0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``dzz`` and ``dww`` at the nodes ``index`` for the angles ``theta``."""
    out = []
    for direction in _directions(theta):
        plus, minus = _arm(grid, index, direction), _arm(grid, index, -direction)
        w_plus, w_minus = _arm_weights(plus, minus)
        out.append(
            w_plus * (_arm_value(plus, u) - u0) + w_minus * (_arm_value(minus, u) - u0)
        )
    return out[0], out[1]


_CELL = ((0, 0), (1, 0), (0, 1), (1, 1))
"""Index offsets of the four nodes of an interpolation cell, in the order of ``_Arm``."""


def _wide_differences_at_angle(
    grid: MixedGrid, index: np.ndarray, theta: float, u: np.ndarray, u0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_wide_differences` for one angle at every node of ``index``, by shifts.

    An arm that stays in the box ends at the same offset from every node, so
    its bilinear weights are four numbers and its values a weighted sum of
    four shifted copies of the grid values. The nodes whose arms may leave
    the box, within ``sqrt(h) + h`` of it, go through :func:`_wide_differences`.
    """
    n, reach = grid.n, grid.arm / grid.h
    pad = int(np.ceil(reach)) + 1
    values = np.pad(u.reshape(n + 1, n + 1), pad)
    flat = (index[:, 0] - 1) * (n - 1) + index[:, 1] - 1
    out = []
    for direction in _directions(np.asarray(theta)):
        total = -2.0 * u0
        for sign in (1.0, -1.0):
            step = sign * reach * direction
            k = np.floor(step).astype(int)
            fx, fy = step - k
            weights = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
            end = sum(
                w * values[pad + 1 + k[0] + a :, pad + 1 + k[1] + b :][: n - 1, : n - 1]
                for w, (a, b) in zip(weights, _CELL, strict=True)
            )
            total = total + np.ravel(end)[flat]
        out.append(total / grid.arm**2)
    near = np.any((index < reach + 1) | (index > n - reach - 1), axis=1)
    if np.any(near):
        exact = _wide_differences(grid, index[near], np.asarray(theta), u, u0[near])
        for d, e in zip(out, exact, strict=True):
            d[near] = e
    return out[0], out[1]


class _Policy(NamedTuple):
    """The optimal control at every unknown: ``case`` 0 for the wide stencil."""

    value: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    case: np.ndarray


@dataclass(frozen=True, kw_only=True)
class MixedHJB(Scheme):
    """The monotone mixed scheme of Chen, Wan and Lin (2018) on a 2D box.

    At every interior node the discrete equation is the maximum over the
    controls ``(a, theta)`` of the HJB operator, each control with its own
    stencil: the standard 7-point stencil where it is monotone, the
    semi-Lagrangian wide stencil with arms ``sqrt(h)`` elsewhere (module
    docstring); ``wide_search`` sets where the wide-stencil controls
    compete. Each control's operator is linear in ``u`` with an M-matrix,
    so :class:`PolicyIteration
    <maesolvers.classical.iterations.PolicyIteration>` converges from any
    start. The 7-point controls are optimised in closed form, the
    wide-stencil ones over ``angles`` angles, so the scheme is consistent
    when ``angles`` grows with ``n``. Where the optimal control at every
    node is a 7-point one (T1, T2), the scheme is second order; where the
    wide stencil enters, first order at most.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2. The
        paper's ``N`` is ``n``: its tables match ``h = 1/N`` on the unit
        square, although its Section 4 counts ``N`` unknowns per side
        (``benchmarks/chen_wan_lin2018.md``, Finding 1).
    angles
        The number ``M`` of angles ``theta = -pi/4 + k pi / (2 M)``,
        ``k = 0, ..., M - 1``, of the wide-stencil search, at least 1; None
        takes ``n``, the paper's ``M = N``.
    wide_search
        Where the wide-stencil controls compete. ``"outside"``: at the nodes
        where neither 7-point stationary point, eq. (30), lies in its
        region; elsewhere that stationary point maximises its stencil's
        objective over all controls and is taken as the optimum, at a cost
        of ``O(1)``, as the paper's cost estimate says and its tables match.
        ``"everywhere"``: at every node, as its Section 5.2 states.
        ``"only"``: every control takes the wide stencil, with ``a``
        unrestricted: the pure semi-Lagrangian scheme of Feng and Jensen
        (2017), which the paper compares with. ``benchmarks/chen_wan_lin2018.md``,
        Finding 2, compares the three.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``, as in :class:`Standard9
        <maesolvers.classical.schemes.Standard9>`.

    Attributes
    ----------
    domains : tuple of type
        ``(Box,)``; :meth:`supports` also asks for equal sides.
    boundaries : tuple of type
        ``(Dirichlet,)``.
    dims : tuple of int
        ``(2,)``.

    Raises
    ------
    TypeError
        On construction, if ``n``, ``angles`` or ``spline_degree`` is not
        an integer.
    ValueError
        On construction, if a parameter is out of range.
    """

    domains: ClassVar[tuple[type, ...]] = (Box,)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...]] = (2,)

    n: int = 64
    angles: int | None = None
    wide_search: Literal["outside", "everywhere", "only"] = "outside"
    spline_degree: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", integer("n", self.n, 2))
        if self.angles is not None:
            object.__setattr__(self, "angles", integer("angles", self.angles, 1))
        if self.wide_search not in ("outside", "everywhere", "only"):
            raise ValueError(
                "wide_search must be 'outside', 'everywhere' or 'only', "
                f"got {self.wide_search!r}"
            )
        spline_degree = integer("spline_degree", self.spline_degree, 1)
        if spline_degree > 5:
            raise ValueError(
                f"spline_degree must be between 1 and 5, got {spline_degree}"
            )
        object.__setattr__(self, "spline_degree", spline_degree)

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is a 2D box with equal sides and a Dirichlet boundary.

        The 7-point stencils and the rotated wide stencil assume square
        cells, hence the equal sides.

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
        width, height = np.subtract(problem.domain.bbox.hi, problem.domain.bbox.lo)
        return bool(np.isclose(width, height, rtol=1e-12, atol=0.0))

    def discretize(self, problem: Problem) -> MixedGrid:
        """Build the grid, the 7-point operators and the angles of the search.

        Parameters
        ----------
        problem
            A problem the scheme supports.

        Returns
        -------
        MixedGrid

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        ValueError
            If ``f`` is negative or not finite at an interior node.
        """
        if not self.supports(problem):
            raise UnsupportedProblemError(
                f"MixedHJB does not support {problem.name}: it needs a 2D Box "
                "with equal sides and a Dirichlet boundary"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports guarantees it
        n, box, g = self.n, problem.domain.bbox, problem.boundary.g
        x = np.linspace(box.lo[0], box.hi[0], n + 1)
        y = np.linspace(box.lo[1], box.hi[1], n + 1)
        h = float(x[1] - x[0])
        nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
        interior = interior_mask(n).ravel()
        f = problem.f(nodes[interior])
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError(
                f"MixedHJB needs a finite f >= 0 at the interior nodes, and "
                f"{problem.name} has f from {np.min(f)} to {np.max(f)} there"
            )
        dxx, dyy, _ = difference_operators(n, h, h)
        dxy1, dxy2 = _cross_differences(n, h)
        count = n if self.angles is None else self.angles
        return MixedGrid(
            n=n,
            h=h,
            x=x,
            y=y,
            nodes=nodes,
            interior=interior,
            index=np.stack(np.divmod(np.flatnonzero(interior), n + 1), axis=1),
            f=f,
            g=g,
            boundary_values=g(nodes[~interior]),
            dxx=dxx,
            dyy=dyy,
            dxy1=dxy1,
            dxy2=dxy2,
            arm=float(np.sqrt(h)),
            angles=-0.25 * np.pi + 0.5 * np.pi * np.arange(count) / count,
            wide_search=self.wide_search,
            laplace=factorized(sp.csr_matrix(dxx + dyy)[:, interior].tocsc()),
        )

    def rhs(self, grid: MixedGrid) -> np.ndarray:
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

    def _values(self, grid: MixedGrid, v: np.ndarray) -> np.ndarray:
        """Return the values at all nodes, in flat order: ``v`` and the boundary data."""
        u = np.empty(len(grid.nodes))
        u[grid.interior] = v
        u[~grid.interior] = grid.boundary_values
        return u

    def _policy(self, grid: MixedGrid, v: np.ndarray) -> _Policy:
        """Find the optimal control at every unknown and the objective's value there."""
        u = self._values(grid, v)
        rows = np.arange(v.size)
        if grid.wide_search == "only":
            value = np.full(v.size, -np.inf)
            a, theta, case = np.zeros(v.size), np.zeros(v.size), np.zeros(v.size, int)
        else:
            value, a, theta, case, stationary = seven_point_controls(
                grid.dxx @ u, grid.dyy @ u, grid.dxy1 @ u, grid.dxy2 @ u, grid.f
            )
            if grid.wide_search == "outside":
                rows = rows[~stationary]
        index, f = grid.index[rows], grid.f[rows]
        for angle in grid.angles:
            dzz, dww = _wide_differences_at_angle(grid, index, angle, u, v[rows])
            weight = wide_stencil_weight(
                dzz, dww, f, angle, restricted=grid.wide_search != "only"
            )
            wide = wide_stencil_value(dzz, dww, weight, f)
            chosen = wide > value[rows]
            better = rows[chosen]
            value[better], a[better] = wide[chosen], weight[chosen]
            theta[better], case[better] = angle, 0
        return _Policy(value, a, theta, case)

    def policy_system(
        self, grid: MixedGrid, v: np.ndarray
    ) -> tuple[sp.csc_matrix, np.ndarray]:
        """Return the linear system ``A v = F`` of the optimal control at ``v``.

        Chen-Wan-Lin (2018), eq. (22) to (28): at each unknown the control
        that maximises the objective, and the row of its stencil. ``A v - F``
        is the residual at ``v``; ``A`` is an M-matrix.

        Parameters
        ----------
        grid
            From :meth:`discretize`.
        v
            The unknowns.

        Returns
        -------
        tuple
            ``(A, F)``: ``A`` sparse of shape ``(m, m)``, ``F`` of shape
            ``(m,)``.
        """
        policy = self._policy(grid, v)
        a11, a22, a12 = control_coefficients(policy.a, policy.theta)
        seven = policy.case > 0
        full = -(
            sp.diags(np.where(seven, a11, 0.0)) @ grid.dxx
            + sp.diags(np.where(seven, a22, 0.0)) @ grid.dyy
            + sp.diags(np.where(policy.case == 1, 2.0 * a12, 0.0)) @ grid.dxy1
            + sp.diags(np.where(policy.case == 2, 2.0 * a12, 0.0)) @ grid.dxy2
        )
        constant = _hjb_term(policy.a, grid.f)
        rows = np.flatnonzero(~seven)
        if rows.size:
            index = grid.index[rows]
            centre = index[:, 0] * (grid.n + 1) + index[:, 1]
            weight_z = policy.a[rows]
            r, c, values = [rows], [centre], [np.zeros(rows.size)]
            for direction, share in zip(
                _directions(policy.theta[rows]), (weight_z, 1.0 - weight_z), strict=True
            ):
                plus, minus = (
                    _arm(grid, index, direction),
                    _arm(grid, index, -direction),
                )
                for arm, w in zip(
                    (plus, minus), _arm_weights(plus, minus), strict=True
                ):
                    coefficient = share * w
                    values[0] += coefficient
                    r.append(np.repeat(rows, 4))
                    c.append(arm.cell.ravel())
                    values.append((-coefficient[:, None] * arm.weights).ravel())
                    constant[rows] -= coefficient * arm.data
            full = full + sp.csr_matrix(
                (np.concatenate(values), (np.concatenate(r), np.concatenate(c))),
                shape=full.shape,
            )
        full = sp.csr_matrix(full)
        constant += full[:, ~grid.interior] @ grid.boundary_values
        return full[:, grid.interior].tocsc(), -constant

    def residual(self, grid: MixedGrid, v: np.ndarray) -> np.ndarray:
        """Return the maximum over the controls of the discrete HJB operator at the unknowns.

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
        matrix, rhs = self.policy_system(grid, v)
        return matrix @ v - rhs

    def jacobian(self, grid: MixedGrid, v: np.ndarray) -> sp.csc_matrix:
        """Return a generalised Jacobian of :meth:`residual`: the optimal control's matrix.

        So :class:`Newton <maesolvers.classical.iterations.Newton>` solves the
        scheme too, as a damped policy iteration.

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
        return self.policy_system(grid, v)[0]

    def solve_laplace(self, grid: MixedGrid, rhs: np.ndarray) -> np.ndarray:
        """Solve the 5-point ``laplace u = rhs`` with the Dirichlet data for the unknowns.

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
        boundary = sp.csr_matrix(grid.dxx + grid.dyy)[:, ~grid.interior]
        boundary = boundary @ grid.boundary_values
        return grid.laplace(rhs - boundary)

    def solution(self, grid: MixedGrid, v: np.ndarray) -> Solution:
        """Node values, the centred Hessian, the spline and the final policy.

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
            the interior nodes (the scheme has no Hessian of its own) and NaN
            at the boundary nodes. ``info`` holds the spacing ``h``;
            ``residual_max``, the max absolute residual; ``wide_fraction``,
            the share of unknowns whose optimal control takes the wide
            stencil; and ``angles``, the ``M`` of the search.
        """
        u = self._values(grid, v)
        n, h = grid.n, grid.h
        dxx, dyy, dxy = difference_operators(n, h, h)
        hess = np.full((len(grid.nodes), 2, 2), np.nan)
        hess[grid.interior, 0, 0] = dxx @ u
        hess[grid.interior, 1, 1] = dyy @ u
        hess[grid.interior, 0, 1] = hess[grid.interior, 1, 0] = dxy @ u
        policy = self._policy(grid, v)
        return Solution(
            u=spline(grid.x, grid.y, u, min(self.spline_degree, n)),
            nodes=grid.nodes,
            values=u,
            node_hessian=hess,
            interior=grid.interior,
            unknowns=int(v.size),
            info={
                "h": (h, h),
                "residual_max": float(np.max(np.abs(policy.value))),
                "wide_fraction": float(np.mean(policy.case == 0)),
                "angles": len(grid.angles),
            },
        )
