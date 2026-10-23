"""Regenerate feng_jensen2017.csv: the semi-Lagrangian scheme on T39 and T40.

Runs FDSemiLagrangian on the two problems of Feng-Jensen, SIAM J. Numer.
Anal. 55 (2017), Section 8, and on T40 turned off the grid (``T40 turned``,
the kink along a line at the angle 0.3 + pi/2, a test-local problem without
a T id; feng_jensen2017.md, Finding 1), over the paper's stencil factors
m = 2, 4, ..., 64 and grids of n = 10 2^l cells over [-1, 1]^2, which hold
about as many nodes as the paper's levels of refinement. Each row is one
solve: the error at the nodes relative to max |u| over the domain, the
paper's measure, and Howard's steps.

With --summary it reads the CSV instead and prints the tables of
feng_jensen2017.md: the smallest relative error over m at each grid, with
its m, next to the paper's Figure 6, and the convergence rates in h and in
the best k, end to end as the paper takes them and by least squares.

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


# Feng-Jensen 2017, Figure 6: the smallest relative Linf error over m at each
# level of refinement, with the minimising m.
PUBLISHED: dict[str, tuple[tuple[int, float, int], ...]] = {
    "T39": (
        (91, 9.26e-2, 2),
        (329, 3.85e-2, 2),
        (1249, 2.15e-2, 2),
        (4865, 9.59e-3, 4),
        (19201, 5.63e-3, 4),
        (76289, 2.44e-3, 8),
        (304129, 1.51e-3, 8),
        (1214465, 6.59e-4, 16),
    ),
    "T40": (
        (91, 1.03e-1, 4),
        (329, 5.69e-2, 4),
        (1249, 3.08e-2, 8),
        (4865, 2.03e-2, 16),
        (19201, 1.38e-2, 32),
        (76289, 9.12e-3, 32),
        (304129, 6.04e-3, 64),
        (1214465, 4.51e-3, 64),
    ),
}


ROUNDING = 1e-12
"""A relative error below this is rounding, and gives no rate."""


def best(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """The row of smallest relative error over m at each grid, coarse to fine."""
    grids = sorted({int(r["n_cells"]) for r in rows if r["problem"] == name})
    return [
        min(
            (r for r in rows if r["problem"] == name and int(r["n_cells"]) == n),
            key=lambda r: float(r["rel_max_err_nodes"]),
        )
        for n in grids
    ]


def rates(x: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    """The order of ``error`` in ``x``: end to end, and the least-squares slope."""
    logs = np.log(x), np.log(error)
    end_to_end = (logs[1][-1] - logs[1][0]) / (logs[0][-1] - logs[0][0])
    return float(end_to_end), float(np.polyfit(*logs, 1)[0])


def summary(rows: list[dict[str, Any]]) -> str:
    """The Markdown tables of feng_jensen2017.md from the rows of the CSV."""
    lines = []
    for name in PROBLEMS:
        chosen = best(rows, name)
        published = PUBLISHED.get(name)
        lines += [f"{name}:", ""]
        head = "| n | nodes | ours | m | steps |"
        rule = "|---|---|---|---|---|"
        if published:
            head += " paper's DoFs | paper | paper's m |"
            rule += "---|---|---|"
        lines += [head, rule]
        for level, r in enumerate(chosen):
            cells = (
                f"| {r['n_cells']} | {r['nodes']} | {float(r['rel_max_err_nodes']):.3g}"
                f" | {float(r['m']):g} | {r['iterations']} |"
            )
            if published and level < len(published):
                dofs, error, m = published[level]
                cells += f" {dofs} | {error:.3g} | {m} |"
            lines.append(cells)
        error = np.array([float(r["rel_max_err_nodes"]) for r in chosen])
        h = np.array([float(r["h"]) for r in chosen])
        k = np.array([float(r["k"]) for r in chosen])
        if np.min(error) < ROUNDING:
            lines += ["", "Errors at the rounding level: no rate."]
        else:
            eh, lh = rates(h, error)
            lines += ["", f"Rates: h^{eh:.2f} end to end (h^{lh:.2f} least squares)"]
            if k[0] != k[-1]:
                ek, lk = rates(k, error)
                lines[-1] += f", k^{ek:.2f} (k^{lk:.2f}) in the best k"
            lines[-1] += "."
        lines.append("")
    return "\n".join(lines)


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
    args = parser.parse_args()
    if args.summary is not None:
        print(summary(read(args.summary)))
        return
    grids = [10 * 2**level for level in range(args.levels)]
    done = read(args.out) if args.out.exists() else []
    finished = {(r["problem"], int(r["n_cells"]), float(r["m"])) for r in done}
    tasks = [
        (name, n, m, not args.no_mlflow)
        for n in reversed(grids)  # the largest first
        for name in PROBLEMS
        for m in FACTORS
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
