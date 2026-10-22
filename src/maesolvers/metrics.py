"""The shared evaluation protocol.

Solution errors are reported both at the method's own nodes and on a fixed
``n x n`` grid over the bounding box (points inside the domain only). The
residual and the convexity statistics use the method's own discrete Hessian.
"""

from __future__ import annotations

import numpy as np

from maesolvers.problems import Problem
from maesolvers.result import Result


def shared_grid(problem: Problem, n: int = 201) -> np.ndarray:
    """Points of the ``n x n`` bounding-box grid that lie in the domain."""
    box = problem.domain.bbox
    x = np.linspace(box.lo[0], box.hi[0], n)
    y = np.linspace(box.lo[1], box.hi[1], n)
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    return pts[problem.domain.contains(pts)]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def solution_errors(result: Result, n: int = 201) -> dict[str, float]:
    """Relative L2 and max errors against the exact solution, if known."""
    exact = result.problem.exact
    if exact is None:
        return {}
    ue = exact(result.nodes)
    out = {
        "rel_l2_nodes": _rel_l2(result.values, ue),
        "max_nodes": float(np.max(np.abs(result.values - ue))),
    }
    pts = shared_grid(result.problem, n)
    uh, ue = result.u(pts), exact(pts)
    out["rel_l2_grid"] = _rel_l2(uh, ue)
    out["max_grid"] = float(np.max(np.abs(uh - ue)))
    return out


def residual(result: Result) -> dict[str, float]:
    """RMS and max of ``det H - f`` where the method defines a Hessian."""
    if result.hessian is None:
        return {}
    ok = np.all(np.isfinite(result.hessian), axis=(-2, -1))
    h = result.hessian[ok]
    r = np.linalg.det(h) - result.problem.f(result.nodes[ok])
    return {
        "residual_rms": float(np.sqrt(np.mean(r**2))),
        "residual_max": float(np.max(np.abs(r))),
    }


def convexity(result: Result) -> dict[str, float]:
    """Smallest Hessian eigenvalue and the fraction of nodes where it is negative."""
    if result.hessian is None:
        return {}
    ok = np.all(np.isfinite(result.hessian), axis=(-2, -1))
    lam = np.linalg.eigvalsh(result.hessian[ok])[..., 0]
    return {
        "eigmin_min": float(np.min(lam)),
        "nonconvex_fraction": float(np.mean(lam < 0)),
    }


def evaluate(result: Result, n: int = 201) -> dict[str, float]:
    """Everything the protocol asks for, in one flat dictionary."""
    out: dict[str, float] = {}
    out.update(solution_errors(result, n))
    out.update(residual(result))
    out.update(convexity(result))
    for key in ("time", "iterations", "unknowns"):
        if key in result.info:
            out[key] = result.info[key]
    return out
