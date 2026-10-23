"""The shared evaluation protocol (ADR 0007).

:func:`evaluate` measures one :class:`~maesolvers.core.result.Result` the same
way for every method; :func:`summarize` aggregates the seeds of one method on
one problem, and :func:`summarize_errors` does the same for errors measured
elsewhere, one per seed by :func:`seed_error`. :func:`gradient_map_error`,
with :func:`gradient_fd`, measures the gradient map of a transport problem
for the reproduction tables that report it, outside :func:`evaluate`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from maesolvers.core._params import positive
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
    """Return the shared evaluation grid of a domain.

    A uniform grid over the bounding box, both ends included on every axis,
    restricted to the domain (test-set specification, Section 2). Only a
    :class:`~maesolvers.core.domains.Box` keeps its boundary points; every
    other domain, a user's own included, keeps its interior points only.

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
    axes = [np.linspace(a, b, size) for a, b in zip(box.lo, box.hi, strict=True)]
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


def gradient_fd(u: Field, points: np.ndarray, step: float | None = None) -> np.ndarray:
    """Gradient of ``u`` by central differences.

    For a gradient the method does not give: of a problem's exact solution,
    or of a result's ``u``. A method's own discrete gradient, such as the
    differences of a grid scheme at its nodes, is passed to
    :func:`gradient_map_error` directly instead.

    ``u`` is evaluated within ``step`` of each point along each axis, so a
    little outside the domain at a boundary point, and must be defined and
    smooth there. A problem's exact solution and a network are. A classical
    result's ``u`` interpolates its grid and is constant past the domain's
    bounding box, so at a boundary point of a ``Box`` the difference is about
    half the derivative: pass the interior points only,
    ``points[domain.interior(points)]``, or the method's own gradient.

    Parameters
    ----------
    u
        The function, defined and smooth within ``step`` of ``points`` along
        each axis.
    points
        Shape ``(..., d)``.
    step
        Difference step; by default ``eps ** (1/3)`` (about ``6e-6``) times
        ``max(1, |x|)`` at each point, which balances the ``O(step^2)``
        truncation error against rounding of order ``eps / step``.

    Returns
    -------
    numpy.ndarray
        Shape ``(..., d)``.
    """
    points = np.asarray(points, dtype=float)
    d = points.shape[-1]
    if step is None:
        scale = np.maximum(1.0, np.linalg.norm(points, axis=-1, keepdims=True))
        h = np.finfo(float).eps ** (1.0 / 3.0) * scale
    else:
        h = np.full((*points.shape[:-1], 1), positive("step", step))
    columns = []
    for i in range(d):
        shift = h * np.eye(d)[i]
        columns.append((u(points + shift) - u(points - shift)) / (2.0 * h[..., 0]))
    return np.stack(columns, axis=-1)


def gradient_map_error(gradient: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Max and RMS of ``|grad u - grad u_exact|`` over some points.

    The error of the transport map that Benamou-Froese-Oberman 2014 (Tables
    1 and 2) and Caboussat-Peruso 2025 (Table 2) report, and
    :func:`evaluate` does not (its errors are of ``u``, ADR 0007). The
    gradient of a transport problem's solution is unique, so no constant is
    subtracted. The pointwise error is the Euclidean norm of the difference.

    Parameters
    ----------
    gradient
        The method's gradient, shape ``(..., d)``: its own discrete gradient
        at its nodes, an autograd gradient, or :func:`gradient_fd` of
        ``result.u`` at points where ``result.u`` is defined a step beyond
        (the interior points, for a grid interpolant; see
        :func:`gradient_fd`).
    exact
        The exact gradient at the same points, shape ``(..., d)``, such as
        :func:`gradient_fd` of ``problem.exact``.

    Returns
    -------
    dict
        ``gradient_max``, the largest pointwise error, and ``gradient_rms``,
        the root mean square over the points. The names do not clash with
        those of :func:`evaluate`, so the two dicts merge. ``gradient_rms``
        is the discrete L2 norm with weights ``1 / M``; the L2 norm over the
        domain is about ``sqrt(volume) * gradient_rms`` for points uniform in
        the domain, which matters where the domain's volume is not 1 (T12's
        unit disk, ``sqrt(pi)``), so a benchmark states which one a paper's
        "L2 error" is. NaN in ``gradient`` gives NaN.

    Raises
    ------
    ValueError
        If the shapes differ or there are no points.
    """
    gradient = np.asarray(gradient, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if gradient.shape != exact.shape or gradient.ndim == 0:
        raise ValueError(
            "gradient and exact need the same shape (..., d), "
            f"got {gradient.shape} and {exact.shape}"
        )
    error = np.linalg.norm(gradient - exact, axis=-1)
    if error.size == 0:
        raise ValueError("gradient_map_error needs at least one point")
    return {
        "gradient_max": float(np.max(error)),
        "gradient_rms": float(np.sqrt(np.mean(error**2))),
    }


def _smallest_eigenvalues(hess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smallest eigenvalue of each finite symmetric matrix, and which were finite."""
    defined = np.all(np.isfinite(hess), axis=(-2, -1))
    return np.linalg.eigvalsh(hess[defined])[..., 0], defined


def _convexity(result: Result) -> dict[str, float]:
    """Non-convex fraction and smallest eigenvalues of the method's Hessian.

    The points are the interior nodes for a node Hessian (``result.interior``,
    else the nodes the domain counts as interior), else the shared grid for a
    callable Hessian. A non-finite Hessian at any of them makes
    ``nonconvex_fraction`` and ``eigmin`` NaN, so a broken result is never
    reported as convex. ``eigmin_exact`` is taken over the points where the
    exact Hessian is finite.
    """
    problem = result.problem
    if result.node_hessian is not None:
        assert result.nodes is not None  # Result guarantees it
        interior = (
            result.interior
            if result.interior is not None
            else problem.domain.interior(result.nodes)
        )
        points, hess = result.nodes[interior], result.node_hessian[interior]
    elif result.hessian is not None:
        points = shared_grid(problem.domain)
        hess = np.asarray(result.hessian(points), dtype=float)
    else:
        return {}
    if len(points) == 0:
        return {}
    eig, defined = _smallest_eigenvalues(hess)
    if np.all(defined):
        out = {
            "nonconvex_fraction": float(np.mean(eig < 0)),
            "eigmin": float(np.min(eig)),
        }
    else:
        out = {"nonconvex_fraction": float("nan"), "eigmin": float("nan")}
    if problem.exact is not None:
        with np.errstate(invalid="ignore"):
            exact_hess = _hessian_fd(problem.exact, points)
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
    Convexity uses the node Hessian at the interior nodes (``result.interior``
    if the method set it, else the nodes inside the domain), else the callable
    Hessian on the shared grid; a non-finite Hessian at any of these points
    gives NaN ``nonconvex_fraction`` and ``eigmin``.

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
        nodes, values = result.nodes, result.values
        if nodes is not None and values is not None:
            out["max"], out["rms"] = _errors(values, exact(nodes), up_to_constant)
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


def seed_error(result: Result | None, key: str = "max") -> float | None:
    """Return the error of one seed as :func:`summarize` counts it, None if it failed.

    A seed fails when it has no result (it timed out or crashed), when its
    run stopped as ``"diverged"`` (``info["stop"]``, which
    :class:`~maesolvers.core.result.Result` defines for every solver; ADR
    0007 names divergence a failure, so the run fails even with a finite
    error, such as a neural run that returns the network from before a
    non-finite loss), or when its error is NaN or infinite. A run that stops
    unconverged with a finite error, at its step budget or because a step
    changed nothing, does not fail.

    Parameters
    ----------
    result
        The seed's result, ``None`` for a seed that gave no result.
    key
        The :func:`evaluate` field to return.

    Returns
    -------
    float or None
        ``evaluate(result)[key]``, or ``None`` for a failed seed.

    Raises
    ------
    KeyError
        If :func:`evaluate` does not report ``key`` for the result, as for a
        problem without an exact solution.
    """
    if result is None or result.info.get("stop") == "diverged":
        return None
    error = float(evaluate(result)[key])
    return error if np.isfinite(error) else None


def summarize_errors(errors: Sequence[float | None], threshold: float) -> Summary:
    """Aggregate one error per seed, measured elsewhere, as :func:`summarize` does.

    For errors measured elsewhere, such as in the worker processes of a
    benchmark script, where the results themselves cannot be sent back.
    Measure each seed with :func:`seed_error`, so that a failed seed, a run
    that stopped as ``"diverged"`` included, comes as ``None``; this function
    cannot see how a run stopped.

    Parameters
    ----------
    errors
        One error per seed; ``None``, NaN or an infinite value for a failed
        seed, which counts as ``+inf``.
    threshold
        A seed succeeds when its error is below this; ``inf`` counts every
        seed with a finite error as a success.

    Returns
    -------
    Summary

    Raises
    ------
    ValueError
        For no seeds, or a threshold that is not positive.
    """
    if len(errors) == 0:
        raise ValueError("summarize needs at least one seed")
    if not threshold > 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    errs = np.array([np.inf if e is None else float(e) for e in errors])
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


def summarize(
    results: Sequence[Result | None], threshold: float, key: str = "max"
) -> Summary:
    """Aggregate the seeds of one method on one problem (ADR 0007).

    Each result is measured with :func:`seed_error`. A failed seed (``None``,
    for a seed that timed out or crashed and so has no result, a run that
    stopped as ``"diverged"``, or a NaN or infinite error) counts with error
    ``+inf`` and is never dropped, so when more than half the seeds fail the
    median is ``inf``. With an even number of seeds the median is the mean of
    the middle two, so exactly half failed also gives ``inf``.

    A run that stops unconverged with a finite error, at its step budget
    (``info["stop"] == "max_steps"`` for a neural solver) or because a step
    changed nothing, is a normal seed, never counted in ``failed``: it
    enters the quantiles at its error, and ``threshold`` decides whether it
    counts in ``success_rate``. Neural training budgets are often spent
    before the gradient tolerance is met, so counting these runs as failed
    would leave no band to report.

    Parameters
    ----------
    results
        One result per seed, ``None`` for a seed that gave no result.
    threshold
        A seed succeeds when its error is below this.
    key
        The :func:`evaluate` field to aggregate.

    Returns
    -------
    Summary

    Raises
    ------
    ValueError
        For no seeds, or a threshold that is not positive.
    KeyError
        If :func:`evaluate` does not report ``key`` for a result, as for a
        problem without an exact solution.
    """
    if len(results) == 0:
        raise ValueError("summarize needs at least one seed")
    return summarize_errors([seed_error(r, key) for r in results], threshold)
