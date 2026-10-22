"""The shared evaluation protocol (ADR 0007).

:func:`evaluate` measures one :class:`~maesolvers.core.result.Result` the same
way for every method; :func:`summarize` aggregates the seeds of one method on
one problem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from maesolvers.core.boundary import Transport
from maesolvers.core.domains import Box, Domain, Field
from maesolvers.core.result import Result

SHARED_GRID_SIZE: Mapping[int, int] = MappingProxyType({2: 201, 3: 51, 4: 21})
"""Points per axis of the shared evaluation grid, by dimension (3D, 4D provisional)."""

_FIELDS = (
    "max",
    "rms",
    "max_grid",
    "rms_grid",
    "nonconvex_fraction",
    "eigmin",
    "eigmin_exact",
    "time",
    "iterations",
    "unknowns",
)
"""Every field :func:`evaluate` can report besides the problem's observables."""


def shared_grid(domain: Domain, size: int | None = None) -> np.ndarray:
    """The shared evaluation grid of a domain.

    A uniform grid over the bounding box, both ends included on every axis,
    restricted to the domain: a box keeps its boundary points, every other
    domain keeps its interior points only (test-set specification, Section 2).

    Parameters
    ----------
    domain
        The domain.
    size
        Points per axis; by default :data:`SHARED_GRID_SIZE` for the dimension.

    Returns
    -------
    numpy.ndarray
        Shape ``(M, d)``.
    """
    d = domain.dim
    if size is None:
        if d not in SHARED_GRID_SIZE:
            raise ValueError(f"no shared grid size for dimension {d}; pass size")
        size = SHARED_GRID_SIZE[d]
    box = domain.bbox
    axes = [np.linspace(a, b, size) for a, b in zip(box.lo, box.hi)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    keep = (
        domain.contains(points) if isinstance(domain, Box) else domain.interior(points)
    )
    return points[keep]


def _errors(
    uh: np.ndarray, u: np.ndarray, up_to_constant: bool = False
) -> tuple[float, float]:
    """Max absolute and RMS error of ``uh`` against ``u``.

    Parameters
    ----------
    uh, u
        Computed and exact values at the same points.
    up_to_constant
        Subtract each side's mean first, for solutions unique up to a constant.

    Returns
    -------
    tuple of float
        ``(max, rms)``; NaN in ``uh`` gives NaN.
    """
    uh, u = np.asarray(uh, dtype=float), np.asarray(u, dtype=float)
    if up_to_constant:
        uh, u = uh - np.mean(uh), u - np.mean(u)
    e = uh - u
    return float(np.max(np.abs(e))), float(np.sqrt(np.mean(e**2)))


def _hessian_fd(u: Field, points: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Hessian of ``u`` by central differences.

    Parameters
    ----------
    u
        The function, evaluated at points within ``step`` of ``points``.
    points
        Shape ``(M, d)``.
    step
        Difference step; the error is ``O(step^2)`` plus rounding of order
        ``eps / step^2``.

    Returns
    -------
    numpy.ndarray
        Shape ``(M, d, d)``, symmetric.
    """
    points = np.asarray(points, dtype=float)
    d = points.shape[-1]
    eye = step * np.eye(d)
    u0 = u(points)
    hess = np.empty((*points.shape[:-1], d, d))
    for i in range(d):
        ei = eye[i]
        hess[..., i, i] = (u(points + ei) - 2.0 * u0 + u(points - ei)) / step**2
        for j in range(i):
            ej = eye[j]
            cross = (
                u(points + ei + ej)
                - u(points + ei - ej)
                - u(points - ei + ej)
                + u(points - ei - ej)
            ) / (4.0 * step**2)
            hess[..., i, j] = hess[..., j, i] = cross
    return hess


def _smallest_eigenvalues(hess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smallest eigenvalue of each finite symmetric matrix, and which were finite."""
    defined = np.all(np.isfinite(hess), axis=(-2, -1))
    return np.linalg.eigvalsh(hess[defined])[..., 0], defined


def _convexity(result: Result) -> dict[str, float]:
    """Non-convex fraction and smallest eigenvalues, where a Hessian is defined.

    The points are the nodes with a finite node Hessian, else the points of
    the shared grid where the callable Hessian is finite.
    """
    if result.node_hessian is not None:
        points, hess = result.nodes, result.node_hessian
    elif result.hessian is not None:
        points = shared_grid(result.problem.domain)
        hess = np.asarray(result.hessian(points), dtype=float)
    else:
        return {}
    eig, defined = _smallest_eigenvalues(hess)
    if eig.size == 0:
        return {}
    out = {
        "nonconvex_fraction": float(np.mean(eig < 0)),
        "eigmin": float(np.min(eig)),
    }
    exact = result.problem.exact
    if exact is not None:
        with np.errstate(invalid="ignore"):
            exact_hess = _hessian_fd(exact, points[defined])
        exact_eig, _ = _smallest_eigenvalues(exact_hess)
        if exact_eig.size:
            out["eigmin_exact"] = float(np.min(exact_eig))
    return out


def evaluate(result: Result) -> dict[str, float]:
    """Measure a result with the shared protocol.

    Errors need the problem's exact solution and are taken at the primary
    points (the result's nodes if it has any, else the shared grid);
    ``max_grid`` and ``rms_grid`` repeat them on the shared grid through
    ``result.u`` when the result has nodes, as a secondary value. For a
    transport problem each side's mean over the points is subtracted first.
    Convexity uses the node Hessian at the nodes where it is defined, else
    the callable Hessian on the shared grid.

    Parameters
    ----------
    result
        Any :class:`~maesolvers.core.result.Result`.

    Returns
    -------
    dict
        Those of ``max``, ``rms``, ``max_grid``, ``rms_grid``,
        ``nonconvex_fraction``, ``eigmin``, ``eigmin_exact``, ``time``,
        ``iterations`` and ``unknowns`` that apply, then the problem's
        observables under their own names.
    """
    problem = result.problem
    out: dict[str, float] = {}
    exact = problem.exact
    up_to_constant = isinstance(problem.boundary, Transport)
    if exact is not None:
        grid = shared_grid(problem.domain)
        if result.nodes is not None:
            out["max"], out["rms"] = _errors(
                result.values, exact(result.nodes), up_to_constant
            )
            out["max_grid"], out["rms_grid"] = _errors(
                result.u(grid), exact(grid), up_to_constant
            )
        else:
            out["max"], out["rms"] = _errors(
                result.u(grid), exact(grid), up_to_constant
            )
    out.update(_convexity(result))
    for key in ("time", "iterations", "unknowns"):
        if key in result.info:
            out[key] = result.info[key]
    for name, observable in problem.observables.items():
        if name in _FIELDS:
            raise ValueError(f"observable {name!r} of {problem.name} shadows a metric")
        out[name] = float(observable(result.u))
    return out


@dataclass(frozen=True)
class Summary:
    """Errors of several seeds of one method on one problem.

    Attributes
    ----------
    median, q05, q95
        Median and 5% and 95% quantiles, linear interpolation between order
        statistics; ``inf`` where failed seeds reach.
    success_rate
        Share of seeds with error below the threshold.
    seeds
        Number of seeds.
    failed
        Number of failed seeds (no result, NaN or infinite error).
    """

    median: float
    q05: float
    q95: float
    success_rate: float
    seeds: int
    failed: int


def _quantile(sorted_values: np.ndarray, q: float) -> float:
    """Linear-interpolation quantile that tolerates ``+inf``.

    numpy's linear quantile returns NaN between a finite value and ``inf``;
    here any interpolation that reaches an infinite order statistic is ``inf``.

    Parameters
    ----------
    sorted_values
        Ascending, non-empty, no NaN.
    q
        In ``[0, 1]``.

    Returns
    -------
    float
    """
    pos = q * (len(sorted_values) - 1)
    lo = int(np.floor(pos))
    frac = float(pos - lo)
    a = float(sorted_values[lo])
    if frac == 0.0:
        return a
    b = float(sorted_values[lo + 1])
    if np.isinf(b):
        return float("inf")
    return a + frac * (b - a)


def summarize(
    metrics: Sequence[Mapping[str, float] | None], threshold: float, key: str = "max"
) -> Summary:
    """Aggregate the seeds of one method on one problem (ADR 0007).

    A failed seed (``None``, as for a timeout, or a NaN or infinite error)
    counts with error ``+inf`` and is never dropped, so when more than half
    the seeds fail the median is ``inf``.

    Parameters
    ----------
    metrics
        One :func:`evaluate` output per seed, ``None`` for a seed that gave no
        result.
    threshold
        A seed succeeds when its error is below this.
    key
        The error to aggregate.

    Returns
    -------
    Summary

    Raises
    ------
    KeyError
        If a seed's metrics lack ``key``, as for a problem without an exact
        solution.
    """
    if len(metrics) == 0:
        raise ValueError("summarize needs at least one seed")
    if not threshold > 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    errs = np.array([np.inf if m is None else float(m[key]) for m in metrics])
    errs[~np.isfinite(errs)] = np.inf
    errs.sort()
    return Summary(
        median=_quantile(errs, 0.5),
        q05=_quantile(errs, 0.05),
        q95=_quantile(errs, 0.95),
        success_rate=float(np.mean(errs < threshold)),
        seeds=len(errs),
        failed=int(np.sum(np.isinf(errs))),
    )
