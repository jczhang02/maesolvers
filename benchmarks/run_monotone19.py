"""Regenerate monotone19.csv and monotone19_iterates.csv.

Runs FDMonotone with the 19-point stencil on the three 3D examples of
Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 4: T9 (the C^2
example (14)), the C^1 example (15), which has no T id and which this module
defines as ``C1_3D``, and T22 (the example with blow-up (16)), at the
printed N = 7, 11, 15, 21 and 31, each with h = 1/N (ADR 0001) and with
h = 1/(N - 1). Then it runs Newton from the paper's start (one semi-implicit
step, laplace u = (d! f)^(1/d), SINUM eq. (8), solved on the grid itself) for
1 to 8 steps at every N, as monotone17.md, Finding 2 does in 2D.

Every solve is one MLflow run in maesolvers-monotone19-<problem> (T9, C1-3D,
T22) unless --no-mlflow is given. The converged solves run in --processes
worker processes (default 8), the iterates in one process per problem, so set
one thread each. Run it on a100server from the repository root::

    just remote-bg monotone19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benchmarks/run_monotone19.py
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from maesolvers import (
    T9,
    T22,
    Box,
    Dirichlet,
    FDMonotone,
    Monotone17,
    Newton,
    Problem,
    Result,
    Solver,
    tracking,
)
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
SINUM_N = (7, 11, 15, 21, 31)
CONVENTIONS = ("1/N", "1/(N-1)")
ITERATE_STEPS = range(1, 9)
CENTRE = np.array([0.5, 0.5, 0.5])


def _c1_exact(x: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(np.asarray(x, dtype=float) - CENTRE, axis=-1)
    return 0.5 * np.maximum(r - 0.2, 0.0) ** 2


def _c1_f(x: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(np.asarray(x, dtype=float) - CENTRE, axis=-1)
    return (1.0 - 0.2 / np.maximum(r, 0.2)) ** 2


C1_3D = Problem(
    "C1-3D",
    Box((0.0,) * 3, (1.0,) * 3),
    _c1_f,
    Dirichlet(_c1_exact),
    exact=_c1_exact,
    note=(
        "Froese-Oberman 2011 (SIAM J. Numer. Anal.), example (15): "
        "u = ((|x - x0| - 0.2)^+)^2 / 2 on (0, 1)^3, x0 = (0.5, 0.5, 0.5), "
        "f = (1 - 0.2/|x - x0|)^2 = 1 - 0.4/|x - x0| + 0.04/|x - x0|^2 outside "
        "the ball of radius 0.2 and 0 inside: T20 in 3D. No T id in the survey "
        "or the shared test-set specification, so not in the catalogue."
    ),
)
"""The 3D C^1 example (15), a test-local problem (issue #27)."""

PROBLEMS: dict[str, Problem] = {"T9": T9, "C1-3D": C1_3D, "T22": T22}

# SINUM Table 4 as printed: max error (mantissa, power of ten), Newton
# iterations and CPU seconds, in the order of SINUM_N.
SINUM: dict[str, tuple[tuple[str, int], ...]] = {
    "T9": tuple((m, 0) for m in ("0.0151", "0.0140", "0.0132", "0.0127", "0.0125")),
    "C1-3D": tuple((m, 0) for m in ("0.0034", "0.0022", "0.0019", "0.0020", "0.0019")),
    "T22": tuple((m, -3) for m in ("9.6", "5.3", "4.7", "4.3", "3.9")),
}
SINUM_ITERATIONS: dict[str, tuple[int, ...]] = {
    "T9": (2, 3, 5, 6, 5),
    "C1-3D": (1, 1, 1, 2, 2),
    "T22": (1, 3, 3, 6, 8),
}
SINUM_CPU: dict[str, tuple[str, ...]] = {
    "T9": ("0.1", "0.1", "0.5", "3.6", "34.7"),
    "C1-3D": ("0.02", "0.06", "0.17", "1.42", "16.70"),
    "T22": ("0.02", "0.12", "0.34", "3.65", "54.70"),
}


def printed(value: float, published: tuple[str, int]) -> str:
    """Our value at the published precision."""
    mantissa, power = published
    decimals = len(mantissa.partition(".")[2])
    return f"{value / 10.0**power:.{decimals}f}"


def published_value(published: tuple[str, int]) -> str:
    """The published value as a float string."""
    return repr(float(published[0]) * 10.0 ** published[1])


def cells(N: int, convention: str) -> int:
    """Cells per side for the printed N under a grid convention."""
    return N if convention == "1/N" else N - 1


@dataclass(frozen=True, kw_only=True)
class PapersStartNewton(Newton):
    """Newton from the paper's start, laplace u = (d! f)^(1/d) (SINUM, eq. (8)).

    Froese and Oberman solve it on a coarse grid, convexify it where needed
    and interpolate; this one solves it on the grid itself.
    """

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """Solve ``laplace u = (d! f)^(1/d)``."""
        scheme: Any = self.scheme
        d = problem.dim
        return scheme.solve_laplace(
            context, (math.factorial(d) * scheme.rhs(context)) ** (1.0 / d)
        )


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[dict[str, Any], Result]:
    """Solve and log to MLflow if ``log``; ``evaluate`` with the info, and the result."""
    if not log:
        result = solver.solve(problem)
        return {**evaluate(result), **result.info}, result
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-monotone19-{problem.name}", run_name=run_name):
        result = solver.solve(problem, callbacks=[logger])
    assert logger.evaluation is not None  # set when the solve ends
    return {**logger.evaluation, **result.info}, result


def sequence(task: tuple[str, int, str, bool]) -> dict[str, Any]:
    """The row of monotone19.csv for one problem, printed N and grid convention."""
    name, N, convention, log = task
    problem, n = PROBLEMS[name], cells(N, convention)
    run_name = f"N={N} h={convention} stencil=19"
    m, _ = solve(FDMonotone(n=n, stencil=19), problem, run_name, log)
    k = SINUM_N.index(N)
    published = SINUM[name][k]
    print(f"{name} {run_name}: max {m['max']:.4g} its {m['iterations']}", flush=True)
    return {
        "problem": name,
        "N": N,
        "h_convention": convention,
        "n_cells": n,
        "unknowns": m["unknowns"],
        "max_err_nodes": m["max"],
        "published": published_value(published),
        "ours_as_printed": printed(m["max"], published),
        "iterations": m["iterations"],
        "published_iterations": SINUM_ITERATIONS[name][k],
        "converged": m["converged"],
        "residual_max": m["residual_max"],
        "nonconvex_fraction": m["nonconvex_fraction"],
        "max_err_grid": m["max_grid"],
        "time_s": f"{m['time']:.2f}",
        "published_cpu_s": SINUM_CPU[name][k],
    }


def iterates(task: tuple[str, bool]) -> list[dict[str, Any]]:
    """The rows of monotone19_iterates.csv for one problem."""
    name, log = task
    problem = PROBLEMS[name]
    rows = []
    for N in SINUM_N:
        for convention in CONVENTIONS:
            n = cells(N, convention)
            for steps in ITERATE_STEPS:
                solver = PapersStartNewton(
                    scheme=Monotone17(n=n, stencil=19), max_iter=steps
                )
                run_name = f"N={N} h={convention} papers start, {steps} steps"
                m, _ = solve(solver, problem, run_name, log)
                rows.append(
                    {
                        "problem": name,
                        "N": N,
                        "h_convention": convention,
                        "n_cells": n,
                        "steps": steps,
                        "iterations": m["iterations"],
                        "max_err_nodes": m["max"],
                        "residual_max": m["residual_max"],
                        "converged": m["converged"],
                    }
                )
                if m["converged"]:
                    break  # further steps change nothing
    print(f"{name} iterates done", flush=True)
    return rows


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV, the header from the first row."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the command line, run the solves, write both CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=8, help="worker processes")
    args = parser.parse_args()
    log = not args.no_mlflow
    tasks = [
        (name, N, convention, log)
        for name in PROBLEMS
        for N in SINUM_N
        for convention in CONVENTIONS
    ]
    tasks.sort(key=lambda task: -task[1])  # the slowest first
    with Pool(args.processes) as pool:
        rows = pool.map(sequence, tasks, 1)
    with Pool(len(PROBLEMS)) as pool:
        iterate_rows = pool.map(iterates, [(name, log) for name in PROBLEMS], 1)
    # Write only after every solve, so no run sees a dirty checkout.
    order = {name: i for i, name in enumerate(PROBLEMS)}
    rows.sort(key=lambda r: (order[r["problem"]], r["h_convention"], r["N"]))
    write(HERE / "monotone19.csv", rows)
    write(
        HERE / "monotone19_iterates.csv",
        [row for chunk in iterate_rows for row in chunk],
    )


if __name__ == "__main__":
    main()
