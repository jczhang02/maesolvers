"""Regenerate hybrid.csv, or hybrid_checks.csv.

Runs FDHybrid on T1, T2 and T20 over the grid sizes of Froese-Oberman,
J. Comput. Phys. 230 (2011), Table 3, with both boundary treatments of the
monotone part, on T2 also with h = 1/(N - 1) (monotone17.md, Finding 3), and
on the cone T21, which the catalogue lacks; run_monotone17.py builds it.

With --checks it writes hybrid_checks.csv instead, the evidence for the
design choices and the findings of hybrid.md: Section 8.1 of the paper
(Newton on the standard and on the hybrid scheme, from the exact solution,
with and without the line search, and from the Poisson start, on T2); the
width of the monotone band on T1 and T20; and the error after each Newton
step on T1 at the coarse grids, against the paper's iteration counts
(Table 1).

Every solve is one MLflow run in maesolvers-hybrid-<problem> unless
--no-mlflow is given. The solves run in --processes worker processes
(default 12), so set one thread each. Run it on a100server from the
repository root::

    just remote-bg hybrid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_hybrid.py
    just remote-bg hybrid-checks env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_hybrid.py --checks
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from run_monotone17 import JCP_N, cone, printed, published_value, write
from scipy.sparse.linalg import spsolve

from maesolvers import (
    T1,
    T2,
    T20,
    FDHybrid,
    Hybrid,
    Monotone17,
    Newton,
    Problem,
    Result,
    Solver,
    Standard9,
    tracking,
)
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
BOUNDARIES = ("narrow", "interpolate")
PROBLEMS = {"T1": T1, "T2": T2, "T20": T20}

# Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Hybrid", as
# printed: (mantissa, power of ten), in the order of JCP_N.
JCP_HYBRID: dict[str, tuple[tuple[str, int], ...]] = {
    "T1": tuple(
        (m, -5)
        for m in ("24.45", "15.29", "9.06", "5.32", "3.02", "1.61", "0.87", "0.46")
    ),
    "T20": tuple(
        (m, -4) for m in ("12.2", "5.9", "4.2", "2.6", "2.0", "1.2", "1.0", "0.7")
    ),
    "T2": tuple(
        (m, -3)
        for m in ("1.74", "0.98", "0.59", "0.35", "0.20", "0.12", "0.07", "0.04")
    ),
    "T21": tuple((m, -3) for m in ("3", "3", "3", "4", "4", "4", "4", "4")),
}


@dataclass(frozen=True, kw_only=True)
class ExactStart(Newton):
    """Newton from the exact solution at the unknowns (JCP 2011, Section 8.1)."""

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """The exact solution at the scheme's unknowns."""
        grid = context.monotone if isinstance(self.scheme, Hybrid) else context
        assert problem.exact is not None
        return problem.exact(grid.nodes[grid.interior])


@dataclass(frozen=True, kw_only=True)
class UndampedExactStart(ExactStart):
    """Full Newton steps, no line search, from the exact solution."""

    def step(
        self, problem: Problem, context: Any, iterate: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """One full Newton step; its max residual, inf if not finite."""
        scheme: Any = self.scheme
        r = scheme.residual(context, iterate)
        new = iterate + spsolve(scheme.jacobian(context, iterate), -r)
        norm = float(np.max(np.abs(scheme.residual(context, new))))
        return new, norm if np.isfinite(norm) else float("inf")


@dataclass(frozen=True, kw_only=True)
class MonotoneRootStart(Newton):
    """Newton from the converged root of ``start``, whatever ``max_iter``.

    Newton with a ``start`` solves it with its own ``max_iter``; this one
    solves it to convergence, so that ``max_iter`` caps the steps on the
    hybrid scheme alone.
    """

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """The node values of Newton on ``start`` with its default budget."""
        assert self.start is not None
        result = Newton(scheme=self.start, max_iter=200, tol=self.tol).solve(problem)
        assert result.values is not None and result.interior is not None
        return result.values[result.interior]


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[dict[str, Any], Result]:
    """Solve and log to MLflow if ``log``; ``evaluate`` with the info, and the result."""
    if not log:
        result = solver.solve(problem)
        return {**evaluate(result), **result.info}, result
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-hybrid-{problem.name}", run_name=run_name):
        result = solver.solve(problem, callbacks=[logger])
    assert logger.evaluation is not None  # set when the solve ends
    return {**logger.evaluation, **result.info}, result


def sequence(task: tuple[str, str, str, bool]) -> list[dict[str, Any]]:
    """The rows of hybrid.csv for one problem, boundary and grid convention."""
    name, boundary, convention, log = task
    rows = []
    for N in JCP_N:
        n = N - 1 if convention == "1/(N-1)" else N
        problem = cone(n) if name == "T21" else PROBLEMS[name]
        solver = FDHybrid(n=n, boundary=boundary)
        run_name = f"N={N} h={convention} {boundary}"
        m, _ = solve(solver, problem, run_name, log)
        published = JCP_HYBRID[name][JCP_N.index(N)]
        print(f"{name} {run_name}: max {m['max']:.4g}", flush=True)
        rows.append(
            {
                "problem": name,
                "N": N,
                "h_convention": convention,
                "n_cells": n,
                "boundary": boundary,
                "max_err_nodes": m["max"],
                "published_jcp": published_value(published),
                "ours_as_printed_jcp": printed(m["max"], published),
                "iterations": m["iterations"],
                "converged": m["converged"],
                "residual_max": m["residual_max"],
                "monotone_fraction": m["monotone_fraction"],
                "nonconvex_fraction": m["nonconvex_fraction"],
                "time_s": f"{m['time']:.2f}",
            }
        )
    return rows


MAIN_TASKS: list[tuple[str, str, str]] = [
    *((name, b, "1/N") for name in ("T1", "T2", "T20") for b in BOUNDARIES),
    ("T2", "narrow", "1/(N-1)"),
    # x0 is a node only for even n, so the cone is run with n = N - 1, as in
    # run_monotone17.py; with n = N odd no node carries the mass.
    ("T21", "narrow", "1/(N-1)"),
]
"""(problem, boundary, grid convention) of every sequence of hybrid.csv."""


def checks(task: tuple[str, str, int, Any, bool]) -> dict[str, Any]:
    """One row of hybrid_checks.csv.

    ``task`` is (check, problem, N, variant, log). For "section81" the
    variant names the scheme and the start, for "band" it is (band,
    boundary), for "steps" the number of Newton steps.
    """
    check, name, N, variant, log = task
    problem = PROBLEMS[name]
    band, boundary, start = 1.0, "narrow", ""
    if check == "section81":
        scheme_name, start = variant
        scheme = Standard9(n=N) if scheme_name == "standard" else Hybrid(n=N)
        if start == "exact":
            solver: Solver = ExactStart(scheme=scheme, max_iter=200)
        elif start == "exact, undamped":
            solver = UndampedExactStart(scheme=scheme, max_iter=20)
        elif start == "poisson":
            solver = Newton(scheme=scheme, max_iter=200)
        else:  # "monotone"
            solver = Newton(scheme=scheme, start=Monotone17(n=N), max_iter=200)
    elif check == "band":
        band, boundary = variant
        solver = FDHybrid(n=N, band=band, boundary=boundary)
        start = "monotone"
    else:  # "steps"
        solver = MonotoneRootStart(
            scheme=Hybrid(n=N), start=Monotone17(n=N), max_iter=variant
        )
        start = "monotone"
    scheme_label = type(solver.scheme).__name__  # type: ignore[attr-defined]
    run_name = f"check={check} N={N} scheme={scheme_label} variant={variant}"
    m, _ = solve(solver, problem, run_name, log)
    print(f"{name} {run_name}: max {m['max']:.4g} its {m['iterations']}", flush=True)
    published = JCP_HYBRID[name][JCP_N.index(N)] if N in JCP_N else None
    return {
        "check": check,
        "problem": name,
        "N": N,
        "scheme": scheme_label,
        "start": start,
        "band": band,
        "boundary": boundary,
        "max_err_nodes": m["max"],
        "published_jcp_hybrid": published_value(published),
        "iterations": m["iterations"],
        "converged": m["converged"],
        "residual_max": m["residual_max"],
        "nonconvex_fraction": m["nonconvex_fraction"],
        "monotone_fraction": m.get("monotone_fraction", ""),
        "history": " ".join(f"{r:.3e}" for r in m["history"]),
        "time_s": f"{m['time']:.2f}",
    }


CHECK_TASKS: list[tuple[str, str, int, Any]] = [
    *(
        ("section81", "T2", N, (scheme, start))
        for N in (31, 63, 127)
        for scheme, start in (
            ("standard", "exact"),
            ("standard", "exact, undamped"),
            ("standard", "poisson"),
            ("hybrid", "exact"),
            ("hybrid", "poisson"),
            ("hybrid", "monotone"),
        )
    ),
    *(
        ("band", name, N, (band, boundary))
        for name in ("T1", "T20")
        for N in (31, 63, 127)
        for band in (0.0, 1.0, 2.0, 3.0)
        for boundary in BOUNDARIES
    ),
    *(("steps", "T1", N, steps) for N in (31, 45, 63) for steps in range(1, 5)),
]
"""(check, problem, N, variant) of every row of hybrid_checks.csv."""


def main() -> None:
    """Parse the command line, run the solves, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=12, help="worker processes")
    parser.add_argument(
        "--checks", action="store_true", help="write hybrid_checks.csv only"
    )
    args = parser.parse_args()
    log = not args.no_mlflow
    if args.checks:
        with Pool(args.processes) as pool:
            rows = pool.map(checks, [(*task, log) for task in CHECK_TASKS], 1)
        write(HERE / "hybrid_checks.csv", rows)
        return
    with Pool(args.processes) as pool:
        chunks = pool.map(sequence, [(*task, log) for task in MAIN_TASKS], 1)
    # Written only after every solve, so no run sees a dirty checkout.
    write(HERE / "hybrid.csv", [row for chunk in chunks for row in chunk])


if __name__ == "__main__":
    main()
