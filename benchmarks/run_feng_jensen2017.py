"""Regenerate feng_jensen2017.csv: the semi-Lagrangian scheme on T39 and T40.

Runs FDSemiLagrangian on the two problems of Feng-Jensen, SIAM J. Numer.
Anal. 55 (2017), Section 8, and on T40 turned off the grid (``T40 turned``,
the kink along a line at the angle 0.3 + pi/2, a test-local problem without
a T id; feng_jensen2017.md, Finding 1), over the paper's stencil factors
m = 2, 4, ..., 64 and grids of n = 10 2^l cells over [-1, 1]^2, which hold
about as many nodes as the paper's levels of refinement. Each row is one
solve: the error at the nodes relative to max |u| over the domain, the
paper's Linf measure; the L2 and H1 errors of the piecewise linear
solution relative to the norms of u over the triangulated domain
(``fe_errors``), the paper's other two columns; and Howard's steps.

With --checks it writes feng_jensen2017_checks.csv (or --out) instead, the
evidence for the known limits of the method page, one solve at a time in
this process. ``fit`` rows solve FDSemiLagrangian and compare, at the root,
the second differences along the stencil directions with the least-squares
Hessian that ``node_hessian`` reports: the share of unknowns where every
second difference is at least ``-NONNEGATIVE_TOL``, the share of those
where the fitted Hessian still has an eigenvalue below ``-FIT_TOL``, and
the smallest such eigenvalue, beside ``evaluate``'s ``nonconvex_fraction``
and ``eigmin``; and the smallest eigenvalue of the Hessian of the
maximising frame, ``d1 s1 s1^T + d2 s2 s2^T``, whose eigenvalues are
``d1`` and ``d2``. ``newton`` and ``howard`` rows time
``Newton(scheme=SemiLagrangian(...))``, which factorises by LU, against
the preset, which solves by Krylov methods first. Run it with the thread
count fixed, as below.

With --summary it reads the CSV instead and prints the tables of
feng_jensen2017.md: the smallest relative error over m at each grid, with
its m, next to the paper's Figure 6, and the convergence rates in h and in
the best k, end to end as the paper takes them and by least squares, for
each of the three norms.

The sweep (``sweep``) leaves out the solves in EXCLUDED: T40 at n = 1280,
m = 2, and T40 turned at n = 1280, m = 2 and 4. At these narrow stencils on
the finest grid Howard's Krylov solves take thousands of steps, and their
budget of 10 N steps keeps the LU fallback out of reach (#72): a run at
dde1578 took 2.5 hours for T40 at m = 2 and 3.6 hours for T40 turned at
m = 4, and T40 turned at m = 2 was still running after 13 hours. None of
them can be the best m at its level (feng_jensen2017.md, Receipt), so
``--levels 8`` ends and writes the receipt's 141 rows.

Every solve is one MLflow run in maesolvers-feng_jensen2017-<problem>
unless --no-mlflow is given. The solves run in --processes worker processes
(default 12), each with one thread, the largest grids first; --levels sets
how many grids (default 7, n = 10 to 640). Each row is written to --out
as its solve ends, and a rerun with the same --out solves only the
problems, grids and factors it lacks. Run it on a100server from the
repository root, writing outside the checkout so that the runs do not see
it dirty, then copy the CSV back::

    just remote-bg feng-jensen env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benchmarks/run_feng_jensen2017.py \\
        --levels 8 --out /jczhang02/maesolvers-data/data/feng_jensen2017.csv
    just remote env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
        python benchmarks/run_feng_jensen2017.py --checks \\
        --out /jczhang02/maesolvers-data/data/feng_jensen2017_checks.csv
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from maesolvers import (
    T20,
    T39,
    T40,
    Dirichlet,
    FDSemiLagrangian,
    Newton,
    Problem,
    SemiLagrangian,
    tracking,
)
from maesolvers.classical.semilagrangian import fit_hessian, frame_values
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
FACTORS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
"""The stencil factors m of the paper's Figures 6 to 8."""

EXCLUDED = frozenset(
    {("T40", 1280, 2.0), ("T40 turned", 1280, 2.0), ("T40 turned", 1280, 4.0)}
)
"""Solves (problem, n, m) the sweep leaves out: see the module docstring."""

TURN = 0.3
"""Angle of the normal of the turned kink, in radians; no stencil direction is
parallel to the kink, since 0.3 + pi/2 is not a multiple of pi/64."""

_NORMAL = np.array([np.cos(TURN), np.sin(TURN)])


def _turned(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(x, dtype=float) @ _NORMAL)


T40_TURNED = Problem(
    "T40 turned",
    T40.domain,
    T40.f,
    Dirichlet(_turned),
    exact=_turned,
    note="T40 with the kink along the line through 0 normal to (cos 0.3, sin 0.3).",
)

PROBLEMS = {"T39": T39, "T40": T40, "T40 turned": T40_TURNED}

U_MAX = {"T39": 4.0, "T40": 1.0, "T40 turned": float(np.sum(_NORMAL))}
"""max |u| over the domain, at the corner (1, 1) or on the circle: |x|^4 is 4
at (1, 1), |x1| is 1 at (+-1, 0) and (1, y), and |x . nu| is
cos 0.3 + sin 0.3 at (1, 1), more than 1 on the circle."""


GRADIENT = {
    "T39": lambda x: 4.0 * np.sum(x**2, axis=-1, keepdims=True) * x,
    "T40": lambda x: np.stack([np.sign(x[..., 0]), np.zeros(x.shape[:-1])], -1),
    "T40 turned": lambda x: np.sign(x @ _NORMAL)[..., None] * _NORMAL,
}
"""The gradient of each exact solution: of |x|^4, |x1| and |x . nu|."""

KINK = {"T40": np.array([1.0, 0.0]), "T40 turned": _NORMAL}
"""The normal of the line through 0 where the exact solution has its kink."""

QUADRATURE_POINTS = 5
"""Gauss points per direction of the collapsed rule on each triangle
(:func:`triangle_rule`): exact for polynomials of degree 8, so for the
squared error of |x|^4 against a linear function."""

_TRIANGLES_PER_BLOCK = 2**15
"""Triangles per block of :func:`fe_errors`: some 50 MB of temporaries."""


def triangle_rule(points: int) -> tuple[np.ndarray, np.ndarray]:
    """The collapsed Gauss rule on the reference triangle ``(0,0), (1,0), (0,1)``.

    The Gauss-Legendre rule of ``points`` points on the square, mapped by
    ``(s, t) -> (s, (1 - s) t)`` with its Jacobian ``1 - s`` in the
    weights: exact for polynomials of degree ``2 points - 2``.

    Returns
    -------
    tuple of numpy.ndarray
        The ``points^2`` points, shape ``(points^2, 2)``, and their weights,
        which add up to the area 1/2.
    """
    t, w = np.polynomial.legendre.leggauss(points)
    t, w = (t + 1.0) / 2.0, w / 2.0
    s, r = np.meshgrid(t, t, indexing="ij")
    weights = np.outer(w, w) * (1.0 - s)
    return np.stack([s, (1.0 - s) * r], -1).reshape(-1, 2), weights.ravel()


def split_along_line(
    corners: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split the triangles that the line ``x . normal = 0`` crosses along it.

    A crossed triangle ``(A, B, C)``, with ``A`` alone on its side, becomes
    ``(A, p, q)``, ``(B, C, q)`` and ``(B, q, p)``, with ``p``, ``q`` where
    the line meets ``AB`` and ``AC``; each piece lies on one side.

    Parameters
    ----------
    corners
        Shape ``(T, 3, 2)``: the corners of ``T`` triangles.
    normal
        Shape ``(2,)``.

    Returns
    -------
    tuple of numpy.ndarray
        The corners of the pieces, shape ``(T', 3, 2)``, and the index of the
        triangle each piece comes from, shape ``(T',)``.
    """
    side = corners @ normal
    crossed = (side.min(axis=1) < 0.0) & (side.max(axis=1) > 0.0)
    kept = np.flatnonzero(~crossed)
    cut = np.flatnonzero(crossed)
    positive = side[cut] > 0.0
    # A is the corner alone on its side: the positive one of one, or the
    # other one of two.
    alone = np.where(
        positive.sum(axis=1) == 1, positive.argmax(axis=1), (~positive).argmax(axis=1)
    )
    order = (alone[:, None] + np.arange(3)) % 3
    abc = np.take_along_axis(corners[cut], order[..., None], axis=1)
    s = np.take_along_axis(side[cut], order, axis=1)
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    p = a + (s[:, 0] / (s[:, 0] - s[:, 1]))[:, None] * (b - a)
    q = a + (s[:, 0] / (s[:, 0] - s[:, 2]))[:, None] * (c - a)
    pieces = np.concatenate(
        [
            corners[kept],
            np.stack([a, p, q], 1),
            np.stack([b, c, q], 1),
            np.stack([b, q, p], 1),
        ]
    )
    return pieces, np.concatenate([kept, cut, cut, cut])


def fe_errors(
    nodes: np.ndarray,
    values: np.ndarray,
    exact: Callable[[np.ndarray], np.ndarray],
    gradient: Callable[[np.ndarray], np.ndarray],
    kink: np.ndarray | None = None,
) -> tuple[float, float]:
    """The relative L2 and H1 errors of the piecewise linear solution.

    ``u_h`` is the piecewise linear function with ``values`` at ``nodes`` on
    their Delaunay triangulation, the one the scheme interpolates on, so
    its domain is the polygon ``Omega_h`` the triangles cover. The errors
    are ``||u - u_h|| / ||u||`` in ``L2(Omega_h)`` and in the full norm of
    ``H1(Omega_h)``, ``||v||^2 + ||grad v||^2``, every integral by the rule
    of :func:`triangle_rule` on each triangle. Where ``u`` has a kink
    along a line through 0, the triangles are first split along it
    (:func:`split_along_line`), so that the rule integrates smooth
    functions.

    Parameters
    ----------
    nodes
        Shape ``(N, 2)``.
    values
        Shape ``(N,)``.
    exact, gradient
        ``u`` and its gradient, from points of shape ``(..., 2)`` to shape
        ``(...)`` and ``(..., 2)``.
    kink
        The normal of the line through 0 where ``u`` has a kink, if any.

    Returns
    -------
    tuple of float
        The relative L2 error and the relative H1 error.
    """
    tri = Delaunay(nodes)
    corners = nodes[tri.simplices]
    # u_h is c + g . x on each triangle.
    edges = corners[:, 1:] - corners[:, :1]
    rise = values[tri.simplices[:, 1:]] - values[tri.simplices[:, :1]]
    g = np.linalg.solve(edges, rise[..., None])[..., 0]
    c = values[tri.simplices[:, 0]] - np.sum(g * corners[:, 0], axis=-1)
    parent = np.arange(len(corners))
    if kink is not None:
        corners, parent = split_along_line(corners, kink)
    points, weights = triangle_rule(QUADRATURE_POINTS)
    sums = np.zeros(4)  # ||e||^2, ||grad e||^2, ||u||^2, ||grad u||^2
    for lo in range(0, len(corners), _TRIANGLES_PER_BLOCK):
        block = corners[lo : lo + _TRIANGLES_PER_BLOCK]
        owner = parent[lo : lo + _TRIANGLES_PER_BLOCK]
        span = block[:, 1:] - block[:, :1]
        area = np.abs(np.linalg.det(span))  # twice the area: the rule's map
        x = block[:, None, 0] + points @ span
        w = area[:, None] * weights
        u, du = exact(x), gradient(x)
        e = u - c[owner, None] - np.sum(g[owner, None] * x, axis=-1)
        de = du - g[owner, None]
        sums += [
            np.sum(w * e**2),
            np.sum(w * np.sum(de**2, axis=-1)),
            np.sum(w * u**2),
            np.sum(w * np.sum(du**2, axis=-1)),
        ]
    l2 = np.sqrt(sums[0] / sums[2])
    h1 = np.sqrt((sums[0] + sums[1]) / (sums[2] + sums[3]))
    return float(l2), float(h1)


def sweep(levels: int) -> list[tuple[str, int, float]]:
    """The solves (problem, n, m) of ``levels`` grids, the largest grids first.

    Every problem and stencil factor on ``n = 10 2^l``, ``l < levels``, but
    the solves in :data:`EXCLUDED`.
    """
    grids = [10 * 2**level for level in range(levels)]
    return [
        (name, n, m)
        for n in reversed(grids)
        for name in PROBLEMS
        for m in FACTORS
        if (name, n, m) not in EXCLUDED
    ]


def run(task: tuple[str, int, float, bool]) -> dict[str, Any]:
    """Solve one problem on one grid with one stencil factor; its CSV row."""
    name, n, m, log = task
    problem = PROBLEMS[name]
    solver = FDSemiLagrangian(n=n, stencil_factor=m)
    if log:
        logger = tracking.MLflowLogger()
        experiment = f"maesolvers-feng_jensen2017-{name.replace(' ', '_')}"
        with tracking.run(experiment, run_name=f"n={n} m={m:g}"):
            result = solver.solve(problem, callbacks=[logger])
            rel_l2, rel_h1 = _fe_errors(name, result)
            import mlflow

            mlflow.log_metrics({"rel_l2_err": rel_l2, "rel_h1_err": rel_h1})
        assert logger.evaluation is not None  # set when the solve ends
        metrics = logger.evaluation
    else:
        result = solver.solve(problem)
        metrics = evaluate(result)
        rel_l2, rel_h1 = _fe_errors(name, result)
    h = max(result.info["h"])
    print(f"{name} n={n} m={m:g}: {metrics['max'] / U_MAX[name]:.3e}", flush=True)
    return {
        "problem": name,
        "n_cells": n,
        "h": h,
        "m": m,
        "k": m * h,
        "nodes": len(result.nodes),
        "unknowns": result.info["unknowns"],
        "max_err_nodes": metrics["max"],
        "rel_max_err_nodes": metrics["max"] / U_MAX[name],
        "rms_err_nodes": metrics["rms"],
        "rel_l2_err": rel_l2,
        "rel_h1_err": rel_h1,
        "max_err_grid": metrics["max_grid"],
        "iterations": result.info["iterations"],
        "converged": result.info["converged"],
        "residual_max": result.info["residual_max"],
        "min_reach": result.info["min_reach"],
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        "time_s": f"{result.info['time']:.2f}",
    }


def _fe_errors(name: str, result: Any) -> tuple[float, float]:
    """:func:`fe_errors` of the solution of the problem ``name``."""
    exact = PROBLEMS[name].exact
    assert exact is not None  # every problem here has its solution
    return fe_errors(result.nodes, result.values, exact, GRADIENT[name], KINK.get(name))


NORMS = {"Linf": "rel_max_err_nodes", "L2": "rel_l2_err", "H1": "rel_h1_err"}
"""The CSV column of the relative error in each norm of the paper's Figure 6."""

PUBLISHED_DOFS = (91, 329, 1249, 4865, 19201, 76289, 304129, 1214465)
"""The paper's number of nodes at each level of refinement (Figure 6)."""

# Feng-Jensen 2017, Figure 6: the smallest relative error over m at each level
# of refinement, with the minimising m, in each norm (sources.md, section 7).
PUBLISHED: dict[str, dict[str, tuple[tuple[float, int], ...]]] = {
    "T39": {
        "L2": (
            (6.92e-2, 2),
            (2.99e-2, 2),
            (1.69e-2, 4),
            (7.12e-3, 4),
            (4.18e-3, 8),
            (1.78e-3, 8),
            (1.06e-3, 16),
            (4.82e-4, 16),
        ),
        "Linf": (
            (9.26e-2, 2),
            (3.85e-2, 2),
            (2.15e-2, 2),
            (9.59e-3, 4),
            (5.63e-3, 4),
            (2.44e-3, 8),
            (1.51e-3, 8),
            (6.59e-4, 16),
        ),
        "H1": (
            (1.86e-1, 2),
            (9.08e-2, 2),
            (4.93e-2, 4),
            (2.28e-2, 4),
            (1.28e-2, 8),
            (5.82e-3, 8),
            (3.36e-3, 8),
            (1.59e-3, 16),
        ),
    },
    "T40": {
        "L2": (
            (4.50e-2, 4),
            (1.62e-2, 4),
            (7.11e-3, 8),
            (3.35e-3, 16),
            (1.70e-3, 32),
            (9.63e-4, 32),
            (5.10e-4, 64),
            (3.12e-4, 64),
        ),
        "Linf": (
            (1.03e-1, 4),
            (5.69e-2, 4),
            (3.08e-2, 8),
            (2.03e-2, 16),
            (1.38e-2, 32),
            (9.12e-3, 32),
            (6.04e-3, 64),
            (4.51e-3, 64),
        ),
        "H1": (
            (2.02e-1, 2),
            (1.51e-1, 4),
            (1.21e-1, 8),
            (9.80e-2, 16),
            (7.91e-2, 32),
            (6.33e-2, 32),
            (5.12e-2, 64),
            (4.25e-2, 64),
        ),
    },
}

PUBLISHED_RATES = {
    "T39": {"L2": (1.02, 1.79), "Linf": (1.02, 1.78), "H1": (0.98, 1.72)},
    "T40": {"L2": (1.02, 2.39), "Linf": (0.64, 1.50), "H1": (0.33, 1.22)},
}
"""The paper's orders in h and in k across its seven refinements (p. 711)."""


ROUNDING = 1e-12
"""A relative error below this is rounding, and gives no rate."""


def best(
    rows: list[dict[str, Any]], name: str, column: str = "rel_max_err_nodes"
) -> list[dict[str, Any]]:
    """The row of smallest error in ``column`` over m at each grid, coarse to fine."""
    grids = sorted({int(r["n_cells"]) for r in rows if r["problem"] == name})
    return [
        min(
            (r for r in rows if r["problem"] == name and int(r["n_cells"]) == n),
            key=lambda r: float(r[column]),
        )
        for n in grids
    ]


def rates(x: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    """The order of ``error`` in ``x``: end to end, and the least-squares slope."""
    logs = np.log(x), np.log(error)
    end_to_end = (logs[1][-1] - logs[1][0]) / (logs[0][-1] - logs[0][0])
    return float(end_to_end), float(np.polyfit(*logs, 1)[0])


def _rate_line(norm: str, chosen: list[dict[str, Any]]) -> str:
    """Our orders in h and in the best k of the best errors in ``norm``."""
    error = np.array([float(r[NORMS[norm]]) for r in chosen])
    if np.min(error) < ROUNDING:
        return f"{norm}: errors at the rounding level, no rate."
    h = np.array([float(r["h"]) for r in chosen])
    k = np.array([float(r["k"]) for r in chosen])
    eh, lh = rates(h, error)
    line = f"{norm}: h^{eh:.2f} end to end (h^{lh:.2f} least squares)"
    if k[0] != k[-1]:
        ek, lk = rates(k, error)
        line += f", k^{ek:.2f} (k^{lk:.2f}) in the best k"
    return line + "."


COMPARED_WITH = {"T39": "T39", "T40": "T40", "T40 turned": "T40"}
"""The problem of Figure 6 whose values go next to each of ours: T40's next to
T40 turned (feng_jensen2017.md, Finding 2)."""


def summary(rows: list[dict[str, Any]]) -> str:
    """The Markdown tables of feng_jensen2017.md from the rows of the CSV.

    For each problem, the best Linf error over m at each grid with its m
    and Howard's steps, then the best L2 and H1 errors, each with its own
    m, next to the paper's Figure 6 (:data:`COMPARED_WITH`), and our orders
    in each norm next to the paper's.
    """
    lines = []
    for name in PROBLEMS:
        paper = COMPARED_WITH[name]
        chosen = {norm: best(rows, name, column) for norm, column in NORMS.items()}
        lines += [
            f"{name}:",
            "",
            "| n | nodes | Linf | m | steps | paper's DoFs | paper | paper's m |",
            "|---|---|---|---|---|---|---|---|",
        ]
        # A CSV of fewer grids pairs them with the paper's first levels.
        for r, dofs, (error, m) in zip(
            chosen["Linf"], PUBLISHED_DOFS, PUBLISHED[paper]["Linf"], strict=False
        ):
            lines.append(
                f"| {r['n_cells']} | {r['nodes']} | {float(r['rel_max_err_nodes']):.3g}"
                f" | {float(r['m']):g} | {r['iterations']} | {dofs} | {error:.3g} | {m} |"
            )
        lines += [
            "",
            "| n | L2 | m | paper | paper's m | H1 | m | paper | paper's m |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for level, r in enumerate(chosen["L2"][: len(PUBLISHED_DOFS)]):
            cells = f"| {r['n_cells']} |"
            for norm in ("L2", "H1"):
                ours = chosen[norm][level]
                error, m = PUBLISHED[paper][norm][level]
                cells += (
                    f" {float(ours[NORMS[norm]]):.3g} | {float(ours['m']):g}"
                    f" | {error:.3g} | {m} |"
                )
            lines.append(cells)
        lines += ["", "Rates:", ""]
        for norm in NORMS:
            rate_h, rate_k = PUBLISHED_RATES[paper][norm]
            lines.append(
                f"- {_rate_line(norm, chosen[norm])}"
                f" Paper: h^{rate_h:.2f}, k^{rate_k:.2f}."
            )
        lines.append("")
    return "\n".join(lines)


NONNEGATIVE_TOL = 1e-8
"""A second difference above ``-NONNEGATIVE_TOL`` counts as nonnegative: at a
root of the scheme they are, up to rounding (about 1e-10 on T40)."""

FIT_TOL = 1e-6
"""A fitted eigenvalue below ``-FIT_TOL`` is negative beyond rounding."""

CHECK_TASKS: tuple[tuple[str, str, int, float], ...] = (
    ("fit", "T20", 361, 4.0),
    ("fit", "T39", 320, 8.0),
    ("fit", "T40", 80, 16.0),
    ("fit", "T40", 320, 16.0),
    ("fit", "T40 turned", 320, 16.0),
    ("howard", "T39", 64, 4.0),
    ("newton", "T39", 64, 4.0),
    ("howard", "T39", 128, 8.0),
    ("newton", "T39", 128, 8.0),
)
"""(check, problem, n, m) of every row of feng_jensen2017_checks.csv; T20 at the
comparison table's n = 361 and m = 4."""


def fit_measures(scheme: SemiLagrangian, problem: Problem, v: np.ndarray) -> dict:
    """The second differences at the unknowns ``v`` against their fitted Hessian."""
    grid = scheme.discretize(problem)
    d = scheme.second_differences(grid, v)
    fitted = np.linalg.eigvalsh(fit_hessian(grid.directions, d))[:, 0]
    nonnegative = d.min(axis=0) >= -NONNEGATIVE_TOL
    half = len(grid.directions) // 2
    d1, d2 = d[:half], d[half:]
    best = frame_values(d1, d2, grid.f).argmax(axis=0)
    cells = np.arange(d.shape[1])
    frame = np.minimum(d1[best, cells], d2[best, cells])
    return {
        "min_second_difference": float(d.min()),
        "nonnegative_share": float(np.mean(nonnegative)),
        "fit_negative_share": float(np.mean(fitted[nonnegative] < -FIT_TOL)),
        "fit_eigmin_nonnegative": float(fitted[nonnegative].min()),
        "frame_eigmin": float(frame.min()),
    }


def check(task: tuple[str, str, int, float], log: bool) -> dict[str, Any]:
    """One row of feng_jensen2017_checks.csv; a value that does not apply is empty."""
    kind, name, n, m = task
    problem = {**PROBLEMS, "T20": T20}[name]
    scheme = SemiLagrangian(n=n, stencil_factor=m)
    solver = (
        Newton(scheme=scheme)
        if kind == "newton"
        else FDSemiLagrangian(n=n, stencil_factor=m)
    )
    experiment = f"maesolvers-feng_jensen2017-{name.replace(' ', '_')}"
    with (
        tracking.run(experiment, run_name=f"check={kind} n={n} m={m:g}")
        if log
        else nullcontext()
    ):
        logger = tracking.MLflowLogger() if log else None
        result = solver.solve(problem, callbacks=[logger] if logger else [])
        metrics = evaluate(result)
        measured = (
            fit_measures(scheme, problem, result.values[result.interior])
            if kind == "fit"
            else {}
        )
        if log:
            import mlflow

            mlflow.log_metrics(measured)
    print(f"{kind} {name} n={n} m={m:g}: {result.info['time']:.1f} s", flush=True)
    empty = dict.fromkeys(
        (
            "min_second_difference",
            "nonnegative_share",
            "fit_negative_share",
            "fit_eigmin_nonnegative",
            "frame_eigmin",
        ),
        "",
    )
    return {
        "check": kind,
        "problem": name,
        "n_cells": n,
        "m": m,
        "solver": type(solver).__name__,
        "time_s": f"{result.info['time']:.2f}",
        "iterations": result.info["iterations"],
        "converged": result.info["converged"],
        "max_err_nodes": metrics["max"],
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        "eigmin": metrics["eigmin"],
        **empty,
        **measured,
        "commit": tracking.git_tags()["mlflow.source.git.commit"],
    }


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a CSV this script wrote."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV, the header from the first row."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the command line, run the solves, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=12, help="worker processes")
    parser.add_argument("--levels", type=int, default=7, help="grids n = 10 2^l")
    parser.add_argument("--out", type=Path, default=HERE / "feng_jensen2017.csv")
    parser.add_argument(
        "--summary", type=Path, default=None, help="print the tables of this CSV"
    )
    parser.add_argument(
        "--checks", action="store_true", help="write the checks CSV instead"
    )
    args = parser.parse_args()
    if args.summary is not None:
        print(summary(read(args.summary)))
        return
    if args.checks:
        out = (
            args.out
            if args.out != parser.get_default("out")
            else HERE / "feng_jensen2017_checks.csv"
        )
        write(out, [check(task, not args.no_mlflow) for task in CHECK_TASKS])
        return
    done = read(args.out) if args.out.exists() else []
    finished = {(r["problem"], int(r["n_cells"]), float(r["m"])) for r in done}
    tasks = [
        (name, n, m, not args.no_mlflow)
        for name, n, m in sweep(args.levels)
        if (name, n, m) not in finished
    ]
    rows: list[dict[str, Any]] = list(done)
    with Pool(args.processes) as pool:
        # Each row is written as soon as its solve ends, so a run that is
        # stopped keeps its finished rows, and a rerun with the same --out
        # solves only the missing ones.
        for row in pool.imap_unordered(run, tasks, 1):
            rows.append(row)
            write(args.out, rows)
    rows.sort(
        key=lambda r: (
            list(PROBLEMS).index(r["problem"]),
            int(r["n_cells"]),
            float(r["m"]),
        )
    )
    write(args.out, rows)


if __name__ == "__main__":
    main()
