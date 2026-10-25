"""Regenerate shortley_weller.csv and shortley_weller_trace.csv (issue #1).

No paper prints a table for the standard 9-point scheme with Shortley-Weller
boundaries, so these receipts hold sanity checks, not a reproduction
(shortley_weller.md).

shortley_weller.csv, one row per solve:

- T8 (unit disk, f = 1, g = 0) with FDNewton and FDPoisson from their
  default starts, and with Newton from the concave start
  laplace u = -2 sqrt(f): which root each reaches;
- the observed order of the max error at the nodes with the exact solution
  of T1, exp(|x|^2 / 2), on the disk inscribed in T1's square, on an ellipse
  in the same square, and on T6 (catalogue), with T1 itself on the box for
  scale.

shortley_weller_trace.csv, one row per Newton step: the residual after each
step of FDNewton on T1 (box) at n = 361 and 512 and on the inscribed disk at
n = 512 (Finding 3), and of Newton on the unscaled residual on the ellipse
and T6 at n = 128 (Finding 2).

Each solve is one MLflow run in maesolvers-<solver>-<problem> unless
--no-mlflow is given; the trace runs are named n=<n>-trace. Both files are
written after the last solve, so no run sees a dirty checkout. Run it on
a100server from the repository root::

    just remote-bg shortley-weller env OMP_NUM_THREADS=16 python benchmarks/run_shortley_weller.py
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
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
from maesolvers.classical.schemes import Grid
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


@dataclass(frozen=True, kw_only=True)
class UnscaledStandard9(Standard9):
    """``Standard9`` with every equation unscaled, ``t = 1`` (Finding 2)."""

    def discretize(self, problem: Problem) -> Grid:
        """Build the grid of :class:`Standard9` with a row scale of 1."""
        grid = super().discretize(problem)
        return dataclasses.replace(grid, row_scale=np.ones_like(grid.row_scale))


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
    """One row of shortley_weller.csv per solve."""

    def concave(n: int) -> Solver:
        return ConcaveStartNewton(scheme=Standard9(n=n))

    presets: list[Callable[[int], Solver]] = [
        lambda n: FDNewton(n=n),
        lambda n: FDPoisson(n=n),
    ]
    cases = [(T8, make) for make in (*presets, concave)]
    for problem in (T1_ON_DISK, T1_ON_ELLIPSE, T6, T1):
        cases += [(problem, make) for make in presets]
    for problem, make in cases:
        for n in SIZES:
            solver = make(n)
            result, m = solve(solver, problem, f"n={n}", log)
            assert result.interior is not None
            print(
                f"{type(solver).__name__} {problem.name} n={n}: max {m['max']:.4g}, "
                f"{m['iterations']} steps, converged {result.info['converged']}",
                flush=True,
            )
            yield {
                "problem": problem.name,
                "solver": type(solver).__name__,
                "n": n,
                "hx": result.info["h"][0],
                "hy": result.info["h"][1],
                "unknowns": m["unknowns"],
                "boundary_points": int(np.count_nonzero(~result.interior)),
                "min_step": result.info["min_step"],
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


def trace(log: bool) -> Iterator[dict[str, object]]:
    """One row of shortley_weller_trace.csv per Newton step."""
    cases: list[tuple[Problem, int, bool]] = [
        (T1, 361, True),
        (T1, 512, True),
        (T1_ON_DISK, 512, True),
        (T1_ON_ELLIPSE, 128, False),
        (T6, 128, False),
    ]
    for problem, n, scaled in cases:
        if scaled:
            solver: Newton = FDNewton(n=n)
            run_name = f"n={n}-trace"
        else:
            solver = Newton(scheme=UnscaledStandard9(n=n))
            run_name = f"n={n}-unscaled-trace"
        result, m = solve(solver, problem, run_name, log)
        assert result.nodes is not None and result.interior is not None
        f = problem.f(result.nodes[result.interior])
        tolerance = solver.tol * max(1.0, float(np.max(np.abs(f))))
        print(
            f"trace {problem.name} n={n} scaled={scaled}: {result.info['history']}",
            flush=True,
        )
        for step, residual in enumerate(result.info["history"], start=1):
            yield {
                "problem": problem.name,
                "solver": type(solver).__name__,
                "n": n,
                "row_scaled": scaled,
                "step": step,
                "residual": residual,
                "tolerance": tolerance,
                "converged": result.info["converged"],
                "max_err_nodes": m["max"],
            }


def write(path: Path, table: list[dict[str, object]]) -> None:
    """Write the rows of ``table`` as a CSV with a header."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    """Parse the command line and write both CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    log = not parser.parse_args().no_mlflow
    # Write only after every solve, so no run sees a dirty checkout.
    table, steps = list(rows(log)), list(trace(log))
    write(HERE / "shortley_weller.csv", table)
    write(HERE / "shortley_weller_trace.csv", steps)


if __name__ == "__main__":
    main()
