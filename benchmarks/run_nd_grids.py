"""Regenerate nd_grids.csv: the cost of the n-dimensional grid and its spline (issue #25).

No paper is reproduced here: the receipt measures what a 3D or 4D grid
method costs `evaluate`, so that the interpolant onto the shared evaluation
grid (51^3 points in 3D, 21^4 in 4D, ADR 0007) stays practical (nd_grids.md).

One row per problem, grid and spline degree: T9 (exp(|x|^2/2) on (0, 1)^3)
and T10 (the same on (0, 1)^4) with the exact solution at the nodes of
``BoxGrid(problem.domain, n)``, the tensor spline of the given degree through
them, and the centred Hessian at the interior nodes, as a grid method would
return them. The columns are the time to build the nodes, to fit the spline,
to evaluate it on the shared grid (the best of three) and to run `evaluate`,
the peak resident memory so far, and the errors `evaluate` reports: zero at
the nodes, the interpolation error on the shared grid.

Each row is one MLflow run in maesolvers-nd-grids-<problem>, named
n=<n>-degree=<k>, unless --no-mlflow is given. The CSV is written after the
last run, so no run sees a dirty checkout. Run it on a100server from the
repository root::

    just remote env OMP_NUM_THREADS=16 python benchmarks/run_nd_grids.py
"""

from __future__ import annotations

import argparse
import csv
import resource
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from maesolvers import T9, T10, BoxGrid, Problem, Result, tracking
from maesolvers.classical.grid import centred_hessian
from maesolvers.metrics import evaluate, shared_grid

HERE = Path(__file__).parent
CASES: tuple[tuple[Problem, tuple[int, ...]], ...] = (
    (T9, (8, 16, 32, 64, 128)),
    (T10, (4, 8, 16, 32)),
)
"""Each problem with its grids, in cells per side."""
DEGREES = (1, 3, 5)
"""Spline degrees; 3 is the schemes' default ``spline_degree``."""


def timed(call: Callable[..., Any], *args: Any, repeat: int = 1) -> tuple[Any, float]:
    """The value of ``call(*args)`` and the shortest of ``repeat`` wall times, in seconds."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        value = call(*args)
        best = min(best, time.perf_counter() - start)
    return value, best


def _build(problem: Problem, n: int) -> BoxGrid:
    """The grid of ``n`` cells per side, its nodes and interior mask computed."""
    grid = BoxGrid(problem.domain, n)
    _ = grid.nodes, grid.interior  # computed once, on first access
    return grid


def rows(log: bool) -> Iterator[dict[str, int | float | str]]:
    """One row of nd_grids.csv per problem, grid and degree."""
    for problem, sizes in CASES:
        assert problem.exact is not None
        points = shared_grid(problem.domain)
        for n in sizes:
            grid, build = timed(_build, problem, n)
            values = problem.exact(grid.nodes)
            hessian = centred_hessian(grid, values)
            for degree in (k for k in DEGREES if k <= n):  # a spline needs k + 1 knots
                u, fit = timed(grid.interpolant, values, degree)
                _, at_points = timed(u, points, repeat=3)
                result = Result(
                    problem,
                    u,
                    nodes=grid.nodes,
                    values=values,
                    node_hessian=hessian,
                    interior=grid.interior,
                )
                m, measure = timed(evaluate, result)
                row: dict[str, int | float | str] = {
                    "problem": problem.name,
                    "dim": problem.dim,
                    "n": n,
                    "nodes": len(grid.nodes),
                    "degree": degree,
                    "shared_points": len(points),
                    "build_s": build,
                    "fit_s": fit,
                    "eval_s": at_points,
                    "evaluate_s": measure,
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024,
                    "max": m["max"],
                    "max_grid": m["max_grid"],
                    "rms_grid": m["rms_grid"],
                    "nonconvex_fraction": m["nonconvex_fraction"],
                }
                print(
                    ", ".join(
                        f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in row.items()
                    ),
                    flush=True,
                )
                if log:
                    _log(problem, n, degree, row)
                yield row


def _log(
    problem: Problem, n: int, degree: int, row: dict[str, int | float | str]
) -> None:
    """Log one row as an MLflow run."""
    import mlflow

    params = {"problem": problem.name, "dim": problem.dim, "n": n, "degree": degree}
    experiment = f"maesolvers-nd-grids-{problem.name}"
    with tracking.run(experiment, params, run_name=f"n={n}-degree={degree}"):
        mlflow.log_metrics(
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        )


def main() -> None:
    """Parse the command line and write nd_grids.csv."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    log = not parser.parse_args().no_mlflow
    table = list(rows(log))  # written only after every run
    with (HERE / "nd_grids.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
