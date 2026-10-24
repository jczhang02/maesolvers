"""The catalogue: every problem maesolvers defines, keyed by its T id (ADR 0007).

The ids are those of the research programme's literature survey and of the
shared test-set specification (``docs/test-set.md`` in
jczhang02/monge-ampere-equations), which fixes the definitions and point
values of the shared test set :data:`SHARED_TEST_SET`. The other problems are
in the catalogue only, defined as the survey and the source paper print them.
An id once used is never reused; the P1 to P5 of 0.0.x are retired (P4 is T1,
P5 is T2, P1 is T8 with ``g = 0``, P3 is T12, P2 left the catalogue).

Every Dirichlet problem with an exact solution takes its boundary data from
that solution. A right-hand side that blows up at a point of the closed
domain (T2, T3, T22, T28) returns ``inf`` there. Every field checks that its points
have shape ``(..., d)`` and raises ``ValueError`` otherwise.

T21 of the specification (the cone ``|x - x0|``, whose ``f`` is a Dirac mass at
``x0``) is not in the catalogue: ``Problem.f`` is a function and cannot hold a
measure.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from maesolvers.core._params import points
from maesolvers.core.boundary import Dirichlet, Transport
from maesolvers.core.domains import (
    Box,
    CutQuarterDisk,
    Disk,
    Domain,
    Ellipse,
    Field,
)
from maesolvers.core.problem import Problem

__all__ = [
    "CATALOGUE",
    "SHARED_TEST_SET",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T12",
    "T16",
    "T20",
    "T22",
    "T28",
    "T46",
    "T20c",
]

# Pure numerical kernels on float points of shape (..., d), values of shape (...).
# `_field` checks the shape before a kernel sees the points.


def _r2(x: np.ndarray, center: float | tuple[float, ...] = 0.0) -> np.ndarray:
    """Squared distance of each point from ``center``."""
    return np.sum((x - center) ** 2, axis=-1)


def _field(dim: int, kernel: Field) -> Field:
    """``kernel`` behind a check that its points have shape ``(..., dim)``."""

    @functools.wraps(kernel)
    def field(x: np.ndarray) -> np.ndarray:
        return kernel(points(x, dim))

    return field


def _const(value: float, dim: int) -> Field:
    return _field(dim, lambda x: np.full(x.shape[:-1], float(value)))


def _unit_box(dim: int) -> Box:
    return Box((0.0,) * dim, (1.0,) * dim)


def _dirichlet(name: str, domain: Domain, f: Field, u: Field, note: str) -> Problem:
    """Build the problem with exact solution ``u`` and its trace as Dirichlet data."""
    u = _field(domain.dim, u)
    return Problem(
        name, domain, _field(domain.dim, f), Dirichlet(u), exact=u, note=note
    )


def _exp_solution(name: str, dim: int, note: str) -> Problem:
    """``u = exp(|x|^2/2)`` on ``(0, 1)^d``: ``f = (1 + |x|^2) exp(d |x|^2/2)``."""

    def u(x: np.ndarray) -> np.ndarray:
        return np.exp(0.5 * _r2(x))

    def f(x: np.ndarray) -> np.ndarray:
        r2 = _r2(x)
        return (1.0 + r2) * np.exp(0.5 * dim * r2)

    return _dirichlet(name, _unit_box(dim), f, u, note)


def _sphere_solution(name: str, dim: int, note: str) -> Problem:
    """``u = -sqrt(d - |x|^2)`` on ``(0, 1)^d``: ``f = d (d - |x|^2)^(-(d+2)/2)``.

    The lower hemisphere of radius ``sqrt d`` through the corner ``(1, ..., 1)``,
    where its gradient blows up and ``f`` is infinite.
    """

    def u(x: np.ndarray) -> np.ndarray:
        return -np.sqrt(dim - _r2(x))

    def f(x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return dim * (dim - _r2(x)) ** (-0.5 * (dim + 2))

    return _dirichlet(name, _unit_box(dim), f, u, note)


# T3: f blows up at the corner (0, 0) but is integrable.


def _t3_exact(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(2.0) / 3.0 * _r2(x) ** 0.75


def _t3_f(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return _r2(x) ** -0.5


# T4, T5: f constant outside a disk, 0 inside.


def _flat_disk(
    name: str,
    domain: Box,
    *,
    center: tuple[float, float],
    radius: float,
    density: float,
    height: float,
    note: str,
) -> Problem:
    """``f = density`` outside the disk ``|x - center| <= radius``, 0 inside.

    The C^1 radial Alexandrov solution: ``u = height`` on the disk and, with
    ``a = radius``, ``r = |x - center|``, ``s = sqrt(r^2 - a^2)``,
    ``u = height + sqrt(density)/2 (r s - a^2 ln((r + s)/a))`` beyond, so
    ``u' = sqrt(density) s``, which vanishes on the circle, and
    ``det D^2 u = u'' u' / r = density``. The gradient maps the disk to the
    point 0, so ``f`` carries no singular part.
    """
    a2 = radius**2

    def u(x: np.ndarray) -> np.ndarray:
        r2 = _r2(x, center)
        r, s = np.sqrt(r2), np.sqrt(np.maximum(r2 - a2, 0.0))
        # On the disk s = 0 and r + s <= radius, so both terms vanish.
        log = np.log(np.maximum(r + s, radius) / radius)
        return height + 0.5 * np.sqrt(density) * (r * s - a2 * log)

    def f(x: np.ndarray) -> np.ndarray:
        return np.where(_r2(x, center) > a2, density, 0.0)

    return _dirichlet(name, domain, f, u, note)


# T6: u = 7/10 exp(a) + x^2 - 1/2 with a = (x - 1/2)^2/2 + y^2.


def _t6_exponential(x: np.ndarray) -> np.ndarray:
    return 0.7 * np.exp(0.5 * (x[..., 0] - 0.5) ** 2 + x[..., 1] ** 2)


def _t6_exact(x: np.ndarray) -> np.ndarray:
    return _t6_exponential(x) + x[..., 0] ** 2 - 0.5


def _t6_f(x: np.ndarray) -> np.ndarray:
    # With E = 7/10 exp(a), s = x - 1/2: u_xx = E (s^2 + 1) + 2,
    # u_yy = E (4 y^2 + 2), u_xy = 2 E s y, so
    # det = E^2 (2 s^2 + 4 y^2 + 2) + 4 E (2 y^2 + 1).
    e, s, y = _t6_exponential(x), x[..., 0] - 0.5, x[..., 1]
    return e**2 * (2.0 * s**2 + 4.0 * y**2 + 2.0) + 4.0 * e * (2.0 * y**2 + 1.0)


# T8: unit disk, f = 1, g = 0; the concave (1 - |x|^2)/2 also solves it.


def _t8_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * (_r2(x) - 1.0)


# T12: unit disk onto an ellipse of the same area pi.


def _t12_exact(x: np.ndarray) -> np.ndarray:
    return x[..., 0] ** 2 + 3.5 * x[..., 0] + 0.25 * x[..., 1] ** 2


# T16: Benamou-Froese-Oberman 2014, Section 6.1. With
# q(z) = (-z^2/(8 pi) + 1/(256 pi^3) + 1/(32 pi)) cos(8 pi z)
#        + z sin(8 pi z) / (32 pi^2),
# u = |x|^2/2 + 4 q(x1) q(x2) has the printed gradient
# (x1 + 4 q'(x1) q(x2), x2 + 4 q(x1) q'(x2)).

_K = 8.0 * np.pi


def _q(z: np.ndarray) -> np.ndarray:
    a = -(z**2) / (8.0 * np.pi) + 1.0 / (256.0 * np.pi**3) + 1.0 / (32.0 * np.pi)
    return a * np.cos(_K * z) + z * np.sin(_K * z) / (32.0 * np.pi**2)


def _dq(z: np.ndarray) -> np.ndarray:
    # The cosine terms cancel: q'(z) = (z^2 - 1/4) sin(8 pi z), zero at +-1/2,
    # so the gradient maps each side of the square into itself.
    return (z**2 - 0.25) * np.sin(_K * z)


def _ddq(z: np.ndarray) -> np.ndarray:
    return 2.0 * z * np.sin(_K * z) + _K * (z**2 - 0.25) * np.cos(_K * z)


def _t16_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * _r2(x) + 4.0 * _q(x[..., 0]) * _q(x[..., 1])


def _t16_f(x: np.ndarray) -> np.ndarray:
    q1, q2 = _q(x[..., 0]), _q(x[..., 1])
    d1, d2 = _dq(x[..., 0]), _dq(x[..., 1])
    dd1, dd2 = _ddq(x[..., 0]), _ddq(x[..., 1])
    return (
        1.0 + 4.0 * (dd1 * q2 + q1 * dd2) + 16.0 * (q1 * q2 * dd1 * dd2 - d1**2 * d2**2)
    )


# T20: f = (1 - 0.2/|x - x0|)^+, zero on the disk |x - x0| <= 0.2.

_T20_CENTER = (0.5, 0.5)


def _t20_exact(x: np.ndarray) -> np.ndarray:
    r = np.sqrt(_r2(x, _T20_CENTER))
    return 0.5 * np.maximum(r - 0.2, 0.0) ** 2


def _t20_f(x: np.ndarray) -> np.ndarray:
    r = np.sqrt(_r2(x, _T20_CENTER))
    return 1.0 - 0.2 / np.maximum(r, 0.2)


# T20c: f = (1 - 0.1/|x|)^+ on (-0.5, 0.5)^2, zero on the disk |x| <= 0.1.


def _t20c_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.maximum(np.sqrt(_r2(x)) - 0.1, 0.0) ** 2


def _t20c_f(x: np.ndarray) -> np.ndarray:
    return 1.0 - 0.1 / np.maximum(np.sqrt(_r2(x)), 0.1)


# T28: T3 in 3D, f = |x|^(-3/2) / 16 blows up at the corner 0 but is integrable.


def _t28_exact(x: np.ndarray) -> np.ndarray:
    return _r2(x) ** 0.75 / 3.0


def _t28_f(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return _r2(x) ** -0.75 / 16.0


# T46: u = |x|^2/2 - sum_k sin x_k on (0, pi)^3, so D^2 u = diag(1 + sin x_k).


def _t46_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * _r2(x) - np.sum(np.sin(x), axis=-1)


def _t46_f(x: np.ndarray) -> np.ndarray:
    return np.prod(1.0 + np.sin(x), axis=-1)


# The problems, in id order.

T1 = _exp_solution(
    "T1",
    2,
    "Smooth solution exp(|x|^2/2) on (0, 1)^2, the standard example of the "
    "literature (Froese-Oberman 2011, Table 3).",
)

T2 = _sphere_solution(
    "T2",
    2,
    "u = -sqrt(2 - |x|^2): the gradient blows up at the corner (1, 1), where f "
    "is infinite and not integrable (Froese-Oberman 2011, Table 3).",
)

T3 = _dirichlet(
    "T3",
    _unit_box(2),
    _t3_f,
    _t3_exact,
    "u = (2 sqrt 2 / 3) |x|^(3/2): f = 1/|x| blows up at the corner (0, 0) "
    "but is integrable (Benamou-Froese-Oberman 2010, Table 3).",
)

T4 = _flat_disk(
    "T4",
    _unit_box(2),
    center=(0.5, 0.5),
    radius=0.4,
    density=1.0,
    height=0.08,
    note="Discontinuous f: 1 outside the disk |x - (0.5, 0.5)| <= 0.4, 0 inside "
    "(Benamou-Froese-Oberman 2010, Section 5.2; the papers print the threshold "
    "as 0.16^2 or 0.15^2). The exact solution is the C^1 Alexandrov solution, "
    "0.08 on the disk, and g is its trace. The paper's "
    "max(|x - (0.5, 0.5)|^2 / 2, 0.08) does not solve the problem: its "
    "gradient jumps on the circle, which adds a singular measure of mass "
    "0.16 pi to f (see benchmarks/sources.md).",
)

T5 = _flat_disk(
    "T5",
    Box((0.0, 0.0), (1.5, 1.5)),
    center=(0.0, 0.0),
    radius=1.0,
    density=4.0,
    height=0.0,
    note="Discontinuous f: 0 for |x| <= 1, 4 beyond, with the C^1 Alexandrov "
    "solution u = 0 inside and r sqrt(r^2 - 1) - ln(r + sqrt(r^2 - 1)) "
    "outside (Nystrom-Vestberg 2023, Example 4.5, on (0, 1.5)^2; "
    "Hu et al. 2026 use (0, 1)^2).",
)

T6 = _dirichlet(
    "T6",
    CutQuarterDisk(),
    _t6_f,
    _t6_exact,
    "Asymmetric domain and solution: u = 0.7 exp((x - 1/2)^2/2 + y^2) "
    "+ x^2 - 1/2 on the quarter disk cut by y > x - 1/2 "
    "(Nystrom-Vestberg 2023, Example 4.6, Table 1).",
)

T7 = Problem(
    "T7",
    Box((-1.0, -1.0), (1.0, 1.0)),
    _const(1.0, 2),
    Dirichlet(_const(1.0, 2)),
    observables={"u_origin": lambda u: float(u(np.zeros(2)))},
    note=(
        "f = 1, g = 1 on [-1, 1]^2: no exact solution, no classical solution "
        "at the corners. Reported as u_origin, u(0, 0), which is also the "
        "minimum of the exact u (Benamou-Froese-Oberman 2010, Section 5.3, "
        "Table 7). That table prints the nodal minimum; compare at even n, "
        "where the origin is a grid node (n = N - 1 in its convention)."
    ),
)

T8 = Problem(
    "T8",
    Disk((0.0, 0.0), 1.0),
    _const(1.0, 2),
    Dirichlet(_const(0.0, 2)),
    exact=_field(2, _t8_exact),
    note=(
        "Branch selection on the unit disk: the concave (1 - |x|^2)/2 also "
        "satisfies det D^2 u = 1, u = 0 on the circle (Caboussat et al. 2026, "
        "Section 6.3)."
    ),
)

T9 = _exp_solution("T9", 3, "T1 in 3D: exp(|x|^2/2) on (0, 1)^3.")

T10 = _exp_solution("T10", 4, "T1 in 4D: exp(|x|^2/2) on (0, 1)^4.")

T12 = Problem(
    "T12",
    Disk((0.0, 0.0), 1.0),
    _const(1.0, 2),
    Transport(Ellipse((3.5, 0.0), (2.0, 0.5))),
    exact=_field(2, _t12_exact),
    note=(
        "Second boundary value problem with a linear optimal map: uniform "
        "densities on the unit disk and on the ellipse centred at (3.5, 0) "
        "with semi-axes (2, 0.5), both of area pi (Caboussat-Peruso 2025)."
    ),
)

T16 = Problem(
    "T16",
    Box((-0.5, -0.5), (0.5, 0.5)),
    _field(2, _t16_f),
    Transport(Box((-0.5, -0.5), (0.5, 0.5))),
    exact=_field(2, _t16_exact),
    note=(
        "Second boundary value problem from the square onto itself with an "
        "oscillating source density built from q(z) "
        "(Benamou-Froese-Oberman 2014, Section 6.1, Table 1)."
    ),
)

T20 = _dirichlet(
    "T20",
    _unit_box(2),
    _t20_f,
    _t20_exact,
    "C^1 solution, f = 0 on the disk |x - (0.5, 0.5)| < 0.2: a degenerate "
    "right-hand side (Froese-Oberman 2011).",
)

T20c = _dirichlet(
    "T20c",
    Box((-0.5, -0.5), (0.5, 0.5)),
    _t20c_f,
    _t20c_exact,
    "T20 with radius 0.1 about the origin on (-0.5, 0.5)^2: C^1 solution, f = 0 "
    "on the disk |x| < 0.1 (Chen-Wan-Lin 2018, Example 3). Defined by the "
    "survey, not yet by the shared specification; the survey prints the "
    "domain [-1, 1]^2, the paper (-0.5, 0.5)^2, which this follows.",
)

T22 = _sphere_solution(
    "T22",
    3,
    "T2 in 3D: u = -sqrt(3 - |x|^2) on (0, 1)^3, gradient blow-up at the "
    "corner (1, 1, 1) (Froese-Oberman 2011, Table 4).",
)

T28 = _dirichlet(
    "T28",
    _unit_box(3),
    _t28_f,
    _t28_exact,
    "T3 in 3D: u = |x|^(3/2) / 3, f = |x|^(-3/2) / 16 blows up at the corner "
    "(0, 0, 0) but is integrable (Brenner-Neilan 2012, Example 3; Liu et al. "
    "2017, Example 4.6). Not in the shared test-set specification yet: the id "
    "and the definition are the survey's.",
)

T46 = _dirichlet(
    "T46",
    Box((0.0,) * 3, (np.pi,) * 3),
    _t46_f,
    _t46_exact,
    "Smooth solution u = |x|^2/2 - sin x - sin y - sin z on (0, pi)^3, "
    "f = (1 + sin x)(1 + sin y)(1 + sin z) (Liu et al. 2017, Example 4.5). Not "
    "in the shared test-set specification yet: the id and the definition are "
    "the survey's.",
)

CATALOGUE: Mapping[str, Problem] = MappingProxyType(
    {
        p.name: p
        for p in (
            T1,
            T2,
            T3,
            T4,
            T5,
            T6,
            T7,
            T8,
            T9,
            T10,
            T12,
            T16,
            T20,
            T20c,
            T22,
            T28,
            T46,
        )
    }
)
"""Every catalogue problem by its T id, in id order; read-only."""

SHARED_TEST_SET: tuple[str, ...] = ("T1", "T2", "T20", "T8", "T12", "T9", "T10")
"""Ids of the shared test set, the rows of the comparison table, in its order."""
