"""Regenerate monotone17.csv and monotone17_iterates.csv.

Runs FDMonotone on T1, T2 and T20 over the grid sizes of Froese-Oberman,
J. Comput. Phys. 230 (2011), Table 3 (the SIAM J. Numer. Anal. 49 (2011)
Table 3 sizes are a subset), with the 9-, 17- and 33-point stencils and both
boundary treatments, and on the cone T21, which the catalogue lacks and this
script builds for each grid. Then it runs Newton from the papers' start for
1 to 8 steps on the coarse grids (monotone17.md, Finding 2). Every solve is
one MLflow run in maesolvers-monotone17-<problem> unless --no-mlflow is
given. The solves run in --processes worker processes (default 16), so set
one thread each. Run it on a100server from the repository root::

    just remote-bg monotone17 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_monotone17.py
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from maesolvers import (
    T1,
    T2,
    T20,
    Box,
    Dirichlet,
    FDMonotone,
    Monotone17,
    Newton,
    Problem,
    Solver,
    tracking,
)
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
JCP_N = (31, 45, 63, 89, 127, 181, 255, 361)
SINUM_N = (31, 63, 127, 255, 361)
STENCILS = (9, 17, 33)
BOUNDARIES = ("narrow", "interpolate")
ITERATE_N = (31, 45, 63)
ITERATE_STEPS = range(1, 9)

# Published max errors as printed: (mantissa, power of ten), in the order of
# SINUM_N or JCP_N. SINUM Table 3 has all three stencils; the JCP Table 3
# column "Monotone" is the 17-point stencil.
SINUM: dict[tuple[str, int], tuple[tuple[str, int], ...]] = {
    ("T1", 9): (("17.9", -4), ("16.2", -4), ("15.9", -4), ("15.9", -4), ("15.9", -4)),
    ("T1", 17): (("8.9", -4), ("5.1", -4), ("4.6", -4), ("4.4", -4), ("4.4", -4)),
    ("T1", 33): (("7.0", -4), ("3.1", -4), ("1.8", -4), ("1.5", -4), ("1.5", -4)),
    ("T20", 9): (("3.0", -3), ("2.5", -3), ("2.3", -3), ("2.2", -3), ("2.2", -3)),
    ("T20", 17): (("1.7", -3), ("1.0", -3), ("0.8", -3), ("0.7", -3), ("0.7", -3)),
    ("T20", 33): (("1.5", -3), ("0.6", -3), ("0.3", -3), ("0.3", -3), ("0.3", -3)),
    ("T2", 9): (("1.7", -3), ("0.9", -3), ("0.8", -3), ("0.8", -3), ("0.8", -3)),
    ("T2", 17): (("1.7", -3), ("0.6", -3), ("0.3", -3), ("0.3", -3), ("0.3", -3)),
    ("T2", 33): (("1.7", -3), ("0.6", -3), ("0.2", -3), ("0.2", -3), ("0.2", -3)),
    ("T21", 9): (("12", -3), ("11", -3), ("11", -3), ("11", -3), ("11", -3)),
    ("T21", 17): (("3", -3), ("3", -3), ("4", -3), ("4", -3), ("4", -3)),
    ("T21", 33): (("3", -3), ("2", -3), ("2", -3), ("1", -3), ("1", -3)),
}
JCP: dict[str, tuple[tuple[str, int], ...]] = {
    "T1": tuple(
        (m, -5)
        for m in (
            "89.09",
            "60.50",
            "50.88",
            "47.51",
            "45.53",
            "44.65",
            "44.22",
            "44.00",
        )
    ),
    "T20": tuple(
        (m, -4) for m in ("17.5", "11.6", "9.8", "8.4", "7.9", "7.4", "7.2", "7.0")
    ),
    "T2": tuple(
        (m, -3)
        for m in ("1.74", "0.98", "0.59", "0.37", "0.35", "0.34", "0.33", "0.33")
    ),
    "T21": tuple((m, -3) for m in ("3", "3", "3", "4", "4", "4", "4", "4")),
}


def printed(value: float, published: tuple[str, int] | None) -> str:
    """Our value at the published precision, or "" without a published value."""
    if published is None:
        return ""
    mantissa, power = published
    decimals = len(mantissa.partition(".")[2])
    return f"{value / 10.0**power:.{decimals}f}"


def published_value(published: tuple[str, int] | None) -> str:
    """The published value as a float string, or ""."""
    return "" if published is None else repr(float(published[0]) * 10.0 ** published[1])


def cone(n: int) -> Problem:
    """T21 on (0, 1)^2 with n cells per side: u = |x - x0|, f = pi delta_x0.

    The papers average the Dirac mass over the disk of radius h/2 about x0,
    so f = 4/h^2 at the nodes within h/2 of x0 and 0 elsewhere (SINUM,
    Section 6.1). x0 = (1/2, 1/2) is a node only for even n.
    """
    h = 1.0 / n

    def exact(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(x, dtype=float) - 0.5, axis=-1)

    def f(x: np.ndarray) -> np.ndarray:
        return np.where(exact(x) <= 0.5 * h * (1 + 1e-12), 4.0 / h**2, 0.0)

    return Problem(
        "T21",
        Box((0.0, 0.0), (1.0, 1.0)),
        f,
        Dirichlet(exact),
        exact=exact,
        note="The cone |x - x0|, f = pi delta_x0 averaged over |x - x0| <= h/2.",
    )


@dataclass(frozen=True, kw_only=True)
class PapersStartNewton(Newton):
    """Newton from the papers' start, laplace u = sqrt(2 f) (SINUM, eq. (8)).

    Froese and Oberman solve it on a coarse grid and interpolate; this one
    solves it on the grid itself.
    """

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = sqrt(2 f)``."""
        scheme: Any = self.scheme
        return scheme.solve_laplace(context, np.sqrt(2.0 * scheme.rhs(context)))


def solve(solver: Solver, problem: Problem, run_name: str, log: bool) -> dict[str, Any]:
    """Solve, log to MLflow if ``log``, and return ``evaluate`` with the result's info."""
    if not log:
        result = solver.solve(problem)
        return {**evaluate(result), **result.info}
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-monotone17-{problem.name}", run_name=run_name):
        result = solver.solve(problem, callbacks=[logger])
    assert logger.evaluation is not None  # set when the solve ends
    return {**logger.evaluation, **result.info}


def sequence(task: tuple[str, int, str, bool]) -> list[dict[str, Any]]:
    """The rows of monotone17.csv for one problem, stencil and boundary."""
    name, stencil, boundary, log = task
    rows = []
    if name == "T21":
        # x0 is a node only for even n: n = N - 1 at every N, and n = N, where f
        # vanishes at every node, at the two smallest N as evidence.
        grids = [("1/(N-1)", N, N - 1) for N in JCP_N]
        grids += [("1/N", N, N) for N in JCP_N[:2]]
    else:
        grids = [("1/N", N, N) for N in JCP_N]
    for convention, N, n in grids:
        problem = cone(n) if name == "T21" else {"T1": T1, "T2": T2, "T20": T20}[name]
        solver = FDMonotone(n=n, stencil=stencil, boundary=boundary)
        run_name = f"N={N} h={convention} stencil={stencil} {boundary}"
        m = solve(solver, problem, run_name, log)
        sinum = (
            SINUM[(name, stencil)][SINUM_N.index(N)]
            if N in SINUM_N and (name, stencil) in SINUM
            else None
        )
        jcp = JCP[name][JCP_N.index(N)] if stencil == 17 else None
        print(f"{name} {run_name}: max {m['max']:.4g}", flush=True)
        rows.append(
            {
                "problem": name,
                "N": N,
                "h_convention": convention,
                "n_cells": n,
                "stencil": stencil,
                "boundary": boundary,
                "max_err_nodes": m["max"],
                "published_sinum": published_value(sinum),
                "ours_as_printed_sinum": printed(m["max"], sinum),
                "published_jcp": published_value(jcp),
                "ours_as_printed_jcp": printed(m["max"], jcp),
                "iterations": m["iterations"],
                "converged": m["converged"],
                "residual_max": m["residual_max"],
                "nonconvex_fraction": m["nonconvex_fraction"],
                "time_s": f"{m['time']:.2f}",
            }
        )
    return rows


def iterates(task: tuple[str, int, bool]) -> list[dict[str, Any]]:
    """The rows of monotone17_iterates.csv for one problem and stencil."""
    name, stencil, log = task
    problem = {"T1": T1, "T2": T2}[name]
    rows = []
    for N in ITERATE_N:
        for steps in ITERATE_STEPS:
            solver = PapersStartNewton(
                scheme=Monotone17(n=N, stencil=stencil), max_iter=steps
            )
            run_name = f"N={N} stencil={stencil} papers start, {steps} steps"
            m = solve(solver, problem, run_name, log)
            rows.append(
                {
                    "problem": name,
                    "N": N,
                    "stencil": stencil,
                    "steps": steps,
                    "iterations": m["iterations"],
                    "max_err_nodes": m["max"],
                    "residual_max": m["residual_max"],
                    "converged": m["converged"],
                }
            )
    return rows


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV, the header from the first row."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the command line, run every solve in parallel, write both CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=16, help="worker processes")
    args = parser.parse_args()
    log = not args.no_mlflow
    main_tasks = [
        (name, stencil, boundary, log)
        for name in ("T1", "T2", "T20", "T21")
        for stencil in STENCILS
        for boundary in BOUNDARIES
    ]
    iterate_tasks = [
        (name, stencil, log) for name in ("T1", "T2") for stencil in STENCILS
    ]
    # The slowest tasks first, so the pool stays busy to the end.
    main_tasks.sort(key=lambda task: -task[1])
    with Pool(args.processes) as pool:
        main_rows: Iterator[list[dict[str, Any]]] = pool.imap(sequence, main_tasks)
        iterate_rows = pool.map(iterates, iterate_tasks)
        rows = [row for chunk in main_rows for row in chunk]
    # Write only after every solve, so no run sees a dirty checkout.
    order = {name: i for i, name in enumerate(("T1", "T2", "T20", "T21"))}
    rows.sort(
        key=lambda r: (
            order[r["problem"]],
            r["h_convention"],
            r["boundary"],
            r["stencil"],
            r["N"],
        )
    )
    write(HERE / "monotone17.csv", rows)
    write(
        HERE / "monotone17_iterates.csv",
        [row for chunk in iterate_rows for row in chunk],
    )


if __name__ == "__main__":
    main()
