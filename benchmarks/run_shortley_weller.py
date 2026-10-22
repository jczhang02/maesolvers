"""Regenerate shortley_weller.csv: Standard9 on curved domains (issue #1).

No paper prints a table for the standard 9-point scheme with Shortley-Weller
boundaries, so this receipt holds sanity checks, not a reproduction
(shortley_weller.md):

- T8 (unit disk, f = 1, g = 0) with FDNewton and FDPoisson from their
  default starts, and with Newton from the concave start
  laplace u = -2 sqrt(f): which root each reaches;
- the observed order of the max error at the nodes with the exact solution
  of T1, exp(|x|^2 / 2), on the disk inscribed in T1's square, on an ellipse
  in the same square, and on T6 (catalogue).

Each solve is one MLflow run in maesolvers-<solver>-<problem> unless
--no-mlflow is given. Run it on a100server from the repository root::

    just remote env OMP_NUM_THREADS=16 python benchmarks/run_shortley_weller.py
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from maesolvers import (
    T1,
    T6,
    T8,
    Dirichlet,
    Disk,
    Ellipse,
    FDNewton,
    FDPoisson,
    Newton,
    Problem,
    Result,
    Solver,
    Standard9,
    tracking,
)
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
SIZES = (16, 32, 64, 128, 256, 512)
"""Cells per side of the grid over the bounding box."""

assert T1.exact is not None
T1_ON_DISK = Problem(
    "t1_exact_on_disk",
    Disk((0.5, 0.5), 0.5),
    T1.f,
    Dirichlet(T1.exact),
    exact=T1.exact,
    note="T1's exact solution on the disk inscribed in its square.",
)
T1_ON_ELLIPSE = Problem(
    "t1_exact_on_ellipse",
    Ellipse((0.5, 0.5), (0.5, 0.3)),
    T1.f,
    Dirichlet(T1.exact),
    exact=T1.exact,
    note="T1's exact solution on an ellipse in its square.",
)


@dataclass(frozen=True, kw_only=True)
class ConcaveStartNewton(Newton):
    """Newton from ``laplace u = -2 sqrt(f)``, the concave branch's trace."""

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = -2 sqrt(f)``."""
        scheme = cast(Standard9, self.scheme)
        return scheme.solve_laplace(context, -2.0 * np.sqrt(scheme.rhs(context)))


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[Result, dict[str, float]]:
    """Solve, log to MLflow if ``log``, and evaluate once."""
    if not log:
        result = solver.solve(problem)
        return result, evaluate(result)
    experiment = f"maesolvers-{type(solver).__name__.lower()}-{problem.name}"
    logger = tracking.MLflowLogger()
    with tracking.run(experiment, run_name=run_name):
        result = solver.solve(problem, callbacks=[logger])
    assert logger.evaluation is not None  # set when the solve ends
    return result, logger.evaluation


def branch(result: Result) -> str:
    """``convex`` or ``concave`` if the node values are T8's root of that sign."""
    assert result.nodes is not None and result.values is not None
    assert T8.exact is not None
    exact = T8.exact(result.nodes)
    for name, root in (("convex", exact), ("concave", -exact)):
        if np.max(np.abs(result.values - root)) < 1e-8:
            return name
    return "other"


def rows(log: bool) -> Iterator[dict[str, object]]:
    """One row per solve."""

    def concave(n: int) -> Solver:
        return ConcaveStartNewton(scheme=Standard9(n=n))

    presets: list[Callable[[int], Solver]] = [
        lambda n: FDNewton(n=n),
        lambda n: FDPoisson(n=n),
    ]
    cases = [(T8, make) for make in (*presets, concave)]
    for problem in (T1_ON_DISK, T1_ON_ELLIPSE, T6):
        cases += [(problem, make) for make in presets]
    for problem, make in cases:
        for n in SIZES:
            solver = make(n)
            result, m = solve(solver, problem, f"n={n}", log)
            grid = Standard9(n=n).discretize(problem)
            print(
                f"{type(solver).__name__} {problem.name} n={n}: max {m['max']:.4g}, "
                f"{m['iterations']} steps, converged {result.info['converged']}",
                flush=True,
            )
            yield {
                "problem": problem.name,
                "solver": type(solver).__name__,
                "n": n,
                "h": result.info["h"][0],
                "unknowns": m["unknowns"],
                "boundary_points": int(np.count_nonzero(~grid.interior)),
                "min_step": float(np.min(grid.row_scale)),
                "iterations": m["iterations"],
                "converged": result.info["converged"],
                "branch": branch(result) if problem is T8 else "",
                "max_err_nodes": m["max"],
                "rms_err_nodes": m["rms"],
                "max_err_grid": m["max_grid"],
                "nonconvex_fraction": m["nonconvex_fraction"],
                "eigmin": m["eigmin"],
                "residual_max": result.info["residual_max"],
                "time_s": f"{m['time']:.2f}",
            }


def main() -> None:
    """Parse the command line and write the CSV."""
    parser = argparse.ArgumentParser(description="Regenerate shortley_weller.csv.")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    log = not parser.parse_args().no_mlflow
    # Write only after every solve, so no run sees a dirty checkout.
    table = list(rows(log))
    with (HERE / "shortley_weller.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
