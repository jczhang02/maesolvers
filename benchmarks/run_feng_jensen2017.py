"""Regenerate feng_jensen2017.csv: the semi-Lagrangian scheme on T39 and T40.

Runs FDSemiLagrangian on the two problems of Feng-Jensen, SIAM J. Numer.
Anal. 55 (2017), Section 8, and on T40 turned off the grid (``T40 turned``,
the kink along a line at the angle 0.3 + pi/2, a test-local problem without
a T id; feng_jensen2017.md, Finding 1), over the paper's stencil factors
m = 2, 4, ..., 64 and grids of n = 10 2^l cells over [-1, 1]^2, which hold
about as many nodes as the paper's levels of refinement. Each row is one
solve: the error at the nodes relative to max |u| over the domain, the
paper's measure, and Howard's steps.

Every solve is one MLflow run in maesolvers-feng_jensen2017-<problem>
unless --no-mlflow is given. The solves run in --processes worker processes
(default 12), each with one thread, the largest grids first; --levels sets
how many grids (default 7, n = 10 to 640). Run it on a100server from the
repository root, writing outside the checkout so that the runs do not see
it dirty, then copy the CSV back::

    just remote-bg feng-jensen env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benchmarks/run_feng_jensen2017.py \\
        --out /jczhang02/maesolvers-data/data/feng_jensen2017.csv
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from maesolvers import T39, T40, Dirichlet, FDSemiLagrangian, Problem, tracking
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
FACTORS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
"""The stencil factors m of the paper's Figures 6 to 8."""

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
        assert logger.evaluation is not None  # set when the solve ends
        metrics = logger.evaluation
    else:
        result = solver.solve(problem)
        metrics = evaluate(result)
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
        "max_err_grid": metrics["max_grid"],
        "iterations": result.info["iterations"],
        "converged": result.info["converged"],
        "residual_max": result.info["residual_max"],
        "min_reach": result.info["min_reach"],
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        "time_s": f"{result.info['time']:.2f}",
    }


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
    args = parser.parse_args()
    grids = [10 * 2**level for level in range(args.levels)]
    tasks = [
        (name, n, m, not args.no_mlflow)
        for n in reversed(grids)  # the largest first
        for name in PROBLEMS
        for m in FACTORS
    ]
    with Pool(args.processes) as pool:
        rows = pool.map(run, tasks, 1)
    rows.sort(key=lambda r: (list(PROBLEMS).index(r["problem"]), r["n_cells"], r["m"]))
    write(args.out, rows)


if __name__ == "__main__":
    main()
