"""Geometry and data checks for the second boundary value problem.

A :class:`~maesolvers.core.boundary.Transport` problem asks that ``grad u``
map the domain onto a convex target set ``Y``. Benamou-Froese-Oberman 2014
(Section 2) write this as ``H(grad u) = 0`` on the boundary, with ``H`` the
signed distance to the boundary of ``Y``, and represent ``Y`` by its support
function ``H*(n) = sup_{y in Y} y . n``. This module gives both for the
target sets :class:`~maesolvers.core.domains.Box`,
:class:`~maesolvers.core.domains.Disk` and
:class:`~maesolvers.core.domains.Ellipse`, as pure functions beside the
domains rather than as members of the
:class:`~maesolvers.core.domains.Domain` protocol (ADR 0007), and the mass
balance a solver checks before it solves.

Everything here takes and returns numpy arrays; the torch version of
:func:`signed_distance` for a neural loss is
:func:`maesolvers.neural.functional.signed_distance`.
"""

from __future__ import annotations

import numpy as np

from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Transport
from maesolvers.core.domains import Box, Disk, Domain, Ellipse, Field, _points
from maesolvers.core.problem import Problem

INTEGRABLE: tuple[type, ...] = (Box, Disk, Ellipse)
"""The domains :func:`integrate` and so :func:`check_mass_balance` handle."""

_BISECTION_STEPS = 160
"""Bisection steps of the ellipse foot point: enough to reach adjacent floats."""


def _unsupported(what: str, domain: object) -> TypeError:
    return TypeError(f"{what} knows Box, Disk and Ellipse, got {type(domain).__name__}")


def support_function(target: Domain, n: np.ndarray) -> np.ndarray:
    """Support function ``sup_{y in Y} y . n`` of a convex target set.

    The Legendre-Fenchel transform ``H*`` of Benamou-Froese-Oberman 2014,
    eq. (5), which encodes the target in the discretisation of the transport
    condition. It is positively homogeneous in ``n``, so ``n`` need not be a
    unit vector.

    Parameters
    ----------
    target
        A :class:`Box` of any dimension, a :class:`Disk` or an :class:`Ellipse`.
    n
        Directions of shape ``(..., d)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(...)``.

    Raises
    ------
    TypeError
        For another kind of target.
    """
    n = _points(n, target.dim)
    if isinstance(target, Box):
        lo, hi = np.asarray(target.lo), np.asarray(target.hi)
        return np.sum(np.maximum(lo * n, hi * n), axis=-1)
    if isinstance(target, Disk):
        return n @ np.asarray(target.center) + target.radius * np.linalg.norm(
            n, axis=-1
        )
    if isinstance(target, Ellipse):
        stretched = np.asarray(target.semi_axes) * n
        return n @ np.asarray(target.center) + np.linalg.norm(stretched, axis=-1)
    raise _unsupported("support_function", target)


def closest_boundary_point(
    target: Domain, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest point of the target's boundary and the outward normal there.

    For every point ``y``, inside or outside, ``y - foot`` is parallel to the
    normal, so ``(y - foot) . normal`` is the signed distance
    (:func:`signed_distance`), and the normal is its gradient wherever the
    nearest point is unique. Where it is not (the centre of a disk, points
    inside a box equally near two faces, points on the major axis of an
    ellipse near its centre), one of the nearest points is returned: the one
    on the first nearest face of a box, in the order ``x_0 = lo[0]``,
    ``x_0 = hi[0]``, ``x_1 = lo[1]``, ...; the one in direction ``(1, 0)``
    for the centre of a disk; the one with nonnegative coordinates relative
    to the centre, in the frame of the semi-axes, for an ellipse.

    Parameters
    ----------
    target
        A :class:`Box` of any dimension, a :class:`Disk` or an :class:`Ellipse`.
    y
        Points of shape ``(..., d)``.

    Returns
    -------
    foot, normal : numpy.ndarray
        Both of shape ``(..., d)``.

    Raises
    ------
    TypeError
        For another kind of target.

    Notes
    -----
    The ellipse has no closed form: the nearest point solves a scalar
    equation, found by bisection after reflecting ``y`` into the first
    quadrant (Eberly, "Distance from a point to an ellipse, an ellipsoid, or
    a hyperellipsoid", Geometric Tools, 2013), which stays robust near the
    centre and the major axis, where Newton's method on the angle does not.
    """
    y = _points(y, target.dim)
    if isinstance(target, Box):
        return _box_foot(np.asarray(target.lo), np.asarray(target.hi), y)
    if isinstance(target, Disk):
        return _disk_foot(np.asarray(target.center), target.radius, y)
    if isinstance(target, Ellipse):
        return _ellipse_foot(np.asarray(target.center), np.asarray(target.semi_axes), y)
    raise _unsupported("closest_boundary_point", target)


def signed_distance(target: Domain, y: np.ndarray) -> np.ndarray:
    """Signed distance from each point to the target's boundary.

    Positive outside the target and negative inside, as ``H`` in
    Benamou-Froese-Oberman 2014, eq. (3), so the transport condition reads
    ``signed_distance(target, grad u) = 0`` on the domain's boundary and
    ``<= 0`` inside. Inside the target it is minus the distance to the
    boundary, the opposite sign of
    :meth:`~maesolvers.core.domains.BoundaryDistance.boundary_distance`.
    For a convex target it equals ``max_{|n| = 1} (y . n - H*(n))`` with
    ``H*`` the :func:`support_function` (their Lemma 2.1).

    Parameters
    ----------
    target
        A :class:`Box` of any dimension, a :class:`Disk` or an :class:`Ellipse`.
    y
        Points of shape ``(..., d)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(...)``.

    Raises
    ------
    TypeError
        For another kind of target.
    """
    y = _points(y, target.dim)
    foot, normal = closest_boundary_point(target, y)
    return np.sum((y - foot) * normal, axis=-1)


def _box_foot(
    lo: np.ndarray, hi: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest boundary point and outward normal of the box ``[lo, hi]``."""
    # Inside: move to the nearest face, normal to it. Outside: clip to the
    # box, normal along y - clip.
    gaps = np.stack([y - lo, hi - y], axis=-1).reshape(*y.shape[:-1], -1)
    face = np.argmin(gaps, axis=-1)  # 2 * axis + (0 for lo, 1 for hi)
    axis, upper = np.divmod(face, 2)
    onehot = np.eye(y.shape[-1], dtype=bool)[axis]
    inner_foot = np.where(onehot, np.where(upper[..., None], hi, lo), y)
    inner_normal = np.where(onehot, np.where(upper[..., None], 1.0, -1.0), 0.0)
    clipped = np.clip(y, lo, hi)
    offset = y - clipped
    length = np.linalg.norm(offset, axis=-1, keepdims=True)
    outside = length[..., 0] > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        outer_normal = offset / length
    foot = np.where(outside[..., None], clipped, inner_foot)
    normal = np.where(outside[..., None], outer_normal, inner_normal)
    return foot, normal


def _disk_foot(
    center: np.ndarray, radius: float, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest boundary point and outward normal of a disk."""
    offset = y - center
    length = np.linalg.norm(offset, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normal = np.where(length > 0.0, offset / length, np.array([1.0, 0.0]))
    return center + radius * normal, normal


def _ellipse_foot(
    center: np.ndarray, semi_axes: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest boundary point and outward normal of an axis-aligned ellipse.

    Eberly's method in the frame where the first semi-axis is the longer
    one and the point has nonnegative coordinates; the result is reflected
    and permuted back.
    """
    order = [0, 1] if semi_axes[0] >= semi_axes[1] else [1, 0]
    e0, e1 = semi_axes[order]
    local = (y - center)[..., order]
    sign = np.where(local < 0.0, -1.0, 1.0)
    y0, y1 = np.abs(local[..., 0]), np.abs(local[..., 1])

    # Off the axes: the foot is (r0 y0 / (s + r0), y1 / (s + 1)) e-scaled, with
    # s the root of F(s) = (r0 z0 / (s + r0))^2 + (z1 / (s + 1))^2 - 1 on
    # [z1 - 1, |(r0 z0, z1)| - 1] (outside) or [z1 - 1, 0] (inside); F falls on it.
    r0 = (e0 / e1) ** 2
    z0, z1 = y0 / e0, y1 / e1
    n0 = r0 * z0
    level = z0**2 + z1**2 - 1.0
    low = z1 - 1.0
    high = np.where(level < 0.0, 0.0, np.hypot(n0, z1) - 1.0)
    # On an axis the bracket can shrink to a pole of F (NaN, never selected below).
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(_BISECTION_STEPS):
            s = 0.5 * (low + high)
            f = (n0 / (s + r0)) ** 2 + (z1 / (s + 1.0)) ** 2 - 1.0
            low = np.where(f > 0.0, s, low)
            high = np.where(f < 0.0, s, high)
            low, high = np.where(f == 0.0, s, low), np.where(f == 0.0, s, high)
        s = 0.5 * (low + high)
        x0 = np.where(level == 0.0, y0, r0 * y0 / (s + r0))
        x1 = np.where(level == 0.0, y1, y1 / (s + 1.0))

    # On the minor axis (y0 = 0) the foot is the vertex (0, e1). On the major
    # axis (y1 = 0) it leaves the axis when y0 e0 < e0^2 - e1^2.
    x0 = np.where(y0 == 0.0, 0.0, x0)
    x1 = np.where(y0 == 0.0, e1, x1)
    on_major = (y1 == 0.0) & (y0 > 0.0)
    span = e0**2 - e1**2
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(span > 0.0, e0 * y0 / span, np.inf)
    leaves = on_major & (ratio < 1.0)
    x0 = np.where(on_major, np.where(leaves, e0 * ratio, e0), x0)
    x1 = np.where(
        on_major,
        np.where(leaves, e1 * np.sqrt(np.maximum(1.0 - ratio**2, 0.0)), 0.0),
        x1,
    )

    gradient = np.stack([x0 / e0**2, x1 / e1**2], axis=-1)
    normal = gradient / np.linalg.norm(gradient, axis=-1, keepdims=True)
    foot = np.stack([x0, x1], axis=-1)
    back = np.argsort(order)
    return center + (sign * foot)[..., back], (sign * normal)[..., back]


def volume(domain: Domain) -> float:
    """Area (in 2D) or volume of a domain.

    Parameters
    ----------
    domain
        A :class:`Box` of any dimension, a :class:`Disk` or an :class:`Ellipse`.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        For another kind of domain.
    """
    if isinstance(domain, Box):
        return float(np.prod(np.subtract(domain.hi, domain.lo)))
    if isinstance(domain, Disk):
        return float(np.pi * domain.radius**2)
    if isinstance(domain, Ellipse):
        return float(np.pi * np.prod(domain.semi_axes))
    raise _unsupported("volume", domain)


def integrate(domain: Domain, f: Field, points: int = 64) -> float:
    """Integral of ``f`` over a domain by Gauss quadrature.

    On a box, the tensor Gauss-Legendre rule with ``points`` nodes per axis
    (``points ** d`` evaluations). On a disk or an ellipse, Gauss-Legendre
    with ``points`` nodes in the radius times the trapezoidal rule with
    ``2 * points`` nodes in the angle, which converges spectrally for a
    smooth periodic integrand. ``f`` is never evaluated on the boundary. The
    error is near rounding for a smooth ``f`` (T16's oscillating ``f``
    included), and of the order ``1 / points`` for an ``f`` that jumps.

    Parameters
    ----------
    domain
        A :class:`Box` of any dimension, a :class:`Disk` or an :class:`Ellipse`.
    f
        The integrand, a :data:`Field`.
    points
        Nodes per axis, at least 1.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        For another kind of domain.
    """
    points = integer("points", points, 1)
    nodes, weights = np.polynomial.legendre.leggauss(points)
    if isinstance(domain, Box):
        lo, hi = np.asarray(domain.lo), np.asarray(domain.hi)
        axes = [
            0.5 * (a + b) + 0.5 * (b - a) * nodes for a, b in zip(lo, hi, strict=True)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        w = np.prod(np.meshgrid(*[weights] * domain.dim, indexing="ij"), axis=0)
        return float(np.sum(w * f(grid)) * np.prod(0.5 * (hi - lo)))
    if isinstance(domain, (Disk, Ellipse)):
        center = np.asarray(domain.center)
        if isinstance(domain, Disk):
            scale = np.array([domain.radius, domain.radius])
        else:
            scale = np.asarray(domain.semi_axes)
        r, wr = 0.5 * (nodes + 1.0), 0.5 * weights
        theta = np.linspace(0.0, 2.0 * np.pi, 2 * points, endpoint=False)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        grid = center + scale * r[:, None, None] * circle[None, :, :]
        w = (wr * r)[:, None] * (2.0 * np.pi / len(theta))
        return float(np.sum(w * f(grid)) * np.prod(scale))
    raise _unsupported("integrate", domain)


def check_mass_balance(problem: Problem, rtol: float = 1e-3, points: int = 64) -> None:
    """Raise unless the source mass of a transport problem equals the target's volume.

    The target density of a
    :class:`~maesolvers.core.boundary.Transport` problem is uniform 1, so a
    solution exists only when the integral of ``f`` over the domain equals
    the volume of the target. A solver of transport problems calls this from
    its ``check``, after its support check, so that it never runs on
    unbalanced data.

    Parameters
    ----------
    problem
        A problem with a :class:`~maesolvers.core.boundary.Transport`
        boundary condition.
    rtol
        Largest relative difference allowed, against the target's volume.
        The default leaves room for an ``f`` that jumps, which
        :func:`integrate` resolves to about ``1 / points``.
    points
        Quadrature nodes per axis, passed to :func:`integrate`.

    Raises
    ------
    ValueError
        If the problem's boundary condition is not ``Transport``, or if the
        source mass and the target's volume differ by more than ``rtol``; the
        message gives both.
    TypeError
        If :func:`integrate` does not know the domain or :func:`volume` the
        target.
    """
    rtol = positive("rtol", rtol)
    boundary = problem.boundary
    if not isinstance(boundary, Transport):
        raise ValueError(
            f"{problem.name} has a {type(boundary).__name__} boundary condition, "
            "not Transport: mass balance does not apply"
        )
    mass = integrate(problem.domain, problem.f, points)
    target = volume(boundary.target)
    if not abs(mass - target) <= rtol * target:
        raise ValueError(
            f"{problem.name} is out of mass balance: the integral of f over the "
            f"domain is {mass:.12g}, the target's volume {target:.12g} "
            f"(relative difference above rtol = {rtol:g})"
        )
