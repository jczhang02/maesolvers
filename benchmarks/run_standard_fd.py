"""Regenerate fd_poisson_sequence.csv and fd_newton_sequence.csv.

Runs FDPoisson and FDNewton on P4 and P5 over the grid sizes of
Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, and writes the CSVs
next to this file; standard_fd.md and plot_standard_fd.py read them.
FDNewton also runs with h = 1/(N - 1), the convention ADR 0001 rejects.
Each solve is one MLflow run in maesolvers-<solver>-<problem> unless
--no-mlflow is given. Run it on a100server from the repository root::

    just remote python benchmarks/run_standard_fd.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from collections.abc import Iterator
from pathlib import Path

from maesolvers import P4, P5, FDNewton, FDPoisson, Problem, Result, Solver, tracking
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
SEQUENCE = (31, 45, 63, 89, 127, 181, 255, 361)
POISSON_P4_UP_TO = 127
"""The Poisson receipt for P4 stops here, as it did when it was first recorded."""

PUBLISHED = {
    "P4": (7.14e-5, 3.39e-5, 1.73e-5, 0.87e-5, 0.43e-5, 0.21e-5, 0.11e-5, 0.05e-5),
    "P5": (17.15e-3, 14.59e-3, 12.53e-3, 10.67e-3, 9.00e-3, 7.59e-3, 6.42e-3, 5.41e-3),
}
"""Froese-Oberman 2011, Table 3, column "Standard", in the order of SEQUENCE."""


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[Result, dict[str, float]]:
    """Solve, log to MLflow if ``log``, and evaluate."""
    run: contextlib.AbstractContextManager = contextlib.nullcontext()
    callbacks = []
    if log:
        experiment = f"maesolvers-{type(solver).__name__.lower()}-{problem.name}"
        run = tracking.run(experiment, run_name=run_name)
        callbacks.append(tracking.MLflowLogger())
    with run:
        result = solver.solve(problem, callbacks=callbacks)
    return result, evaluate(result)


def poisson_rows(log: bool) -> Iterator[dict[str, object]]:
    """The rows of fd_poisson_sequence.csv."""
    for problem in (P5, P4):
        for n in SEQUENCE:
            if problem is P4 and n > POISSON_P4_UP_TO:
                continue
            result, m = solve(FDPoisson(n=n), problem, f"N={n}", log)
            print(f"FDPoisson {problem.name} N={n}: max {m['max']:.4g}", flush=True)
            yield {
                "problem": problem.name,
                "N": n,
                "max_err_nodes": m["max"],
                "published_max_err": PUBLISHED[problem.name][SEQUENCE.index(n)],
                "eigmin_min": m["eigmin"],
                "nonconvex_fraction": m["nonconvex_fraction"],
                "iterations": m["iterations"],
                "converged": result.info["converged"],
                "residual_max": result.info["residual_max"],
                "time_s": f"{m['time']:.2f}",
            }


def newton_rows(log: bool) -> Iterator[dict[str, object]]:
    """The rows of fd_newton_sequence.csv, both grid conventions."""
    for problem in (P4, P5):
        for n in SEQUENCE:
            for convention, cells in (("1/N", n), ("1/(N-1)", n - 1)):
                name = f"N={n} h={convention}"
                result, m = solve(FDNewton(n=cells), problem, name, log)
                print(f"FDNewton {problem.name} {name}: max {m['max']:.4g}", flush=True)
                yield {
                    "problem": problem.name,
                    "N": n,
                    "h_convention": convention,
                    "n_cells": cells,
                    "max_err_nodes": m["max"],
                    "eigmin_min": m["eigmin"],
                    "nonconvex_fraction": m["nonconvex_fraction"],
                    "newton_iterations": m["iterations"],
                    "converged": result.info["converged"],
                    "time_s": f"{m['time']:.2f}",
                }


def write(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as CSV, the header from the first row."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the command line and write both CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    log = not parser.parse_args().no_mlflow
    # Write only after every solve, so no run sees a dirty checkout.
    poisson, newton = list(poisson_rows(log)), list(newton_rows(log))
    write(HERE / "fd_poisson_sequence.csv", poisson)
    write(HERE / "fd_newton_sequence.csv", newton)


if __name__ == "__main__":
    main()
