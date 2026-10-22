"""Regenerate filtered.csv, or filtered_checks.csv.

Runs FDFiltered and FDMonotone with the 9-, 17- and 33-point stencils, and
the standard scheme (FDPoisson), on the four examples of Froese-Oberman,
SIAM J. Numer. Anal. 51 (2013), Section 4.2, over the grid sizes of their
Tables 4.1 and 4.2: the C^2 example centred at x0 = (1/2, 1/2), which the
catalogue lacks and this script builds; T20; T2; and the cone T21, built per
grid by run_monotone17.cone. The paper's N counts nodes per side, so the main
rows use n = N - 1 cells (filtered.md, Finding 1), with rows at n = N as
evidence; T1 itself (uncentred) runs at n = N, the catalogue's convention.

For a solve that does not converge, the row also gives the error of the
iterate eight Newton steps before the last (a second solve with a smaller
max_iter), so a reader can judge how settled the last iterate is.

With --checks it writes filtered_checks.csv instead, the evidence for the
findings: the monotone part with kappa = 1 on T2; other filter widths on
T20 and T2; Newton on the filtered scheme from the monotone and the
standard roots on T20 and T2; Newton on the standard scheme alone
(FDNewton) on T20; the filtered scheme with the regularised monotone part
of SINUM 2011, Section 3.5 (run_monotone17.SmoothMonotone17), as FO 2013,
p. 438, says its Newton used, on T2, T1 and T20; and FDFiltered on every
catalogue problem it supports at n = 16 and 64.

Every solve is one MLflow run in maesolvers-filtered-<problem> unless
--no-mlflow is given. The solves run in --processes worker processes
(default 16), so set one thread each. Run it on a100server from the
repository root::

    just remote-bg filtered env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_filtered.py
    just remote-bg filtered-checks env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_filtered.py --checks
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from run_monotone17 import SmoothMonotone17, cone

from maesolvers import (
    T1,
    T2,
    T3,
    T4,
    T5,
    T7,
    T20,
    Box,
    Dirichlet,
    FDFiltered,
    FDMonotone,
    FDNewton,
    FDPoisson,
    Filtered,
    Monotone17,
    Newton,
    Problem,
    Result,
    Solver,
    Standard9,
    tracking,
)
from maesolvers.classical.monotone import MonotoneGrid, angular_resolution
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
SIZES = (31, 63, 127, 255, 361)
STENCILS = (9, 17, 33)
X0 = np.array([0.5, 0.5])


def _centred_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.sum((np.asarray(x, dtype=float) - X0) ** 2, axis=-1) / 2)


def _centred_exp_f(x: np.ndarray) -> np.ndarray:
    r2 = np.sum((np.asarray(x, dtype=float) - X0) ** 2, axis=-1)
    return (1 + r2) * np.exp(r2)


# The same problem as tests/centred.py, which the tests import; a script run
# from the repository root does not see the tests directory.
C2 = Problem(
    "C2-centred",
    Box((0.0, 0.0), (1.0, 1.0)),
    _centred_exp_f,
    Dirichlet(_centred_exp),
    exact=_centred_exp,
    note="FO 2013, eq. (4.1): exp(|x - x0|^2 / 2), x0 = (1/2, 1/2); T1 centred.",
)

# Published max errors as printed, (mantissa, power of ten), in the order of
# SIZES. Table 4.1: (problem, scheme, stencil); Table 4.2: (problem, scheme),
# 17 points. Table 4.2 has no standard value at N = 255 and 361 on T20 and T21.
T41: dict[tuple[str, str, int], tuple[tuple[str, int], ...]] = {}


def _column(power: int, *mantissas: str) -> tuple[tuple[str, int], ...]:
    return tuple((m, power) for m in mantissas)


for _name, _power, _columns in (
    (
        "C2-centred",
        -5,
        (
            ("9.45", "4.91", "3.79", "3.51", "3.48"),
            ("4.54", "1.04", "0.26", "0.06", "0.03"),
            ("9.12", "3.42", "1.67", "1.17", "1.08"),
            ("4.54", "1.04", "0.26", "0.06", "0.03"),
            ("9.38", "3.40", "1.39", "0.66", "0.51"),
            ("4.54", "1.04", "0.26", "0.06", "0.03"),
        ),
    ),
    (
        "T20",
        -4,
        (
            ("21.54", "21.33", "21.55", "21.51", "21.53"),
            ("3.73", "1.51", "0.92", "0.38", "0.23"),
            ("8.66", "6.82", "6.63", "6.58", "6.62"),
            ("3.99", "1.40", "0.76", "0.46", "0.31"),
            ("6.39", "3.18", "2.49", "2.36", "2.37"),
            ("3.67", "1.46", "0.78", "0.37", "0.28"),
        ),
    ),
    (
        "T2",
        -3,
        (
            ("1.74", "0.86", "0.83", "0.83", "0.83"),
            ("1.74", "0.59", "0.20", "0.15", "0.17"),
            ("1.74", "0.59", "0.35", "0.33", "0.33"),
            ("1.74", "0.59", "0.20", "0.13", "0.13"),
            ("1.74", "0.59", "0.20", "0.16", "0.15"),
            ("1.74", "0.59", "0.20", "0.08", "0.08"),
        ),
    ),
    (
        "T21",
        -3,
        (
            ("11.83", "11.10", "11.80", "10.47", "10.40"),
            ("10.42", "11.56", "10.97", "11.03", "10.37"),
            ("3.56", "3.49", "3.45", "3.46", "3.45"),
            ("4.16", "2.82", "2.83", "3.06", "3.22"),
            ("1.61", "1.65", "1.64", "1.64", "1.64"),
            ("1.23", "1.90", "1.11", "1.15", "1.12"),
        ),
    ),
):
    for _i, _column_values in enumerate(_columns):
        _scheme = ("monotone", "filtered")[_i % 2]
        T41[(_name, _scheme, STENCILS[_i // 2])] = _column(_power, *_column_values)

T42: dict[tuple[str, str], tuple[tuple[str, int] | None, ...]] = {
    ("C2-centred", "filtered"): _column(-5, "4.54", "1.06", "0.26", "0.06", "0.03"),
    ("C2-centred", "standard"): _column(-5, "4.54", "1.06", "0.26", "0.06", "0.03"),
    ("T20", "filtered"): _column(-4, "3.99", "1.40", "0.76", "0.46", "0.31"),
    ("T20", "standard"): (*_column(-4, "3.78", "1.34", "0.59"), None, None),
    ("T2", "filtered"): _column(-3, "1.74", "0.59", "0.20", "0.13", "0.13"),
    ("T2", "standard"): _column(-3, "17.38", "12.62", "9.04", "6.43", "5.42"),
    ("T21", "filtered"): _column(-3, "4.16", "2.82", "2.83", "3.06", "3.22"),
    ("T21", "standard"): (*_column(-3, "5.19", "2.86", "1.54"), None, None),
}
T42_ITERATIONS = {
    "C2-centred": (2, 2, 2, 2, 2),
    "T20": (2, 2, 3, 3, 6),
    "T2": (7, 9, 11, 8, 12),
    "T21": (7, 9, 8, 10, 9),
}
"""Table 4.2, Newton iterations of the filtered scheme (17 points)."""


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


CATALOGUE = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T7": T7, "T20": T20}
"""The catalogue problems FDFiltered supports."""


def problem_for(name: str, n: int) -> Problem:
    """The problem of one row; the cone depends on the grid."""
    if name == "T21":
        return cone(n)
    return {"C2-centred": C2, **CATALOGUE}[name]


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[dict[str, Any], Result]:
    """Solve and log to MLflow if ``log``; ``evaluate`` with the info, and the result."""
    if not log:
        result = solver.solve(problem)
        return {**evaluate(result), **result.info}, result
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-filtered-{problem.name}", run_name=run_name):
        result = solver.solve(problem, callbacks=[logger])
    assert logger.evaluation is not None  # set when the solve ends
    return {**logger.evaluation, **result.info}, result


BACK = 8
"""Steps back from the last iterate of a solve that does not converge."""


def make(
    scheme: str, n: int, stencil: int, boundary: str, max_iter: int = 200
) -> Solver:
    """The solver of one row."""
    if scheme == "filtered":
        return FDFiltered(n=n, stencil=stencil, boundary=boundary, max_iter=max_iter)
    if scheme == "monotone":
        return FDMonotone(n=n, stencil=stencil, boundary=boundary)
    return FDPoisson(n=n)


def sequence(task: tuple[str, str, int, str, str, bool]) -> list[dict[str, Any]]:
    """The rows of filtered.csv for one problem, scheme, stencil, boundary, grid."""
    name, scheme, stencil, boundary, convention, log = task
    sizes = SIZES if convention == "1/(N-1)" else SIZES[:2]
    if name == "T1":
        sizes = SIZES
    if scheme == "standard" and name in ("T20", "T21"):
        sizes = sizes[:3]  # Table 4.2 prints no standard value beyond N = 127
    rows = []
    for N in sizes:
        n = N - 1 if convention == "1/(N-1)" else N
        solver = make(scheme, n, stencil, boundary)
        run_name = f"{scheme} N={N} h={convention} stencil={stencil} {boundary}"
        problem = problem_for(name, n)
        m, _ = solve(solver, problem, run_name, log)
        back = ""
        if scheme == "filtered" and not m["converged"]:
            earlier = make(
                scheme, n, stencil, boundary, max_iter=max(m["iterations"] - BACK, 1)
            )
            back = evaluate(earlier.solve(problem))["max"]
        i = SIZES.index(N)
        t41 = T41.get((name, scheme, stencil))
        t42 = T42.get((name, scheme)) if stencil == 17 or scheme == "standard" else None
        p41 = t41[i] if t41 is not None else None
        p42 = t42[i] if t42 is not None else None
        print(
            f"{name} {run_name}: max {m['max']:.4g} its {m['iterations']}", flush=True
        )
        rows.append(
            {
                "problem": name,
                "scheme": scheme,
                "N": N,
                "h_convention": convention,
                "n_cells": n,
                "stencil": stencil if scheme != "standard" else 9,
                "boundary": boundary if scheme != "standard" else "",
                "max_err_nodes": m["max"],
                "published_table41": published_value(p41),
                "ours_as_printed_table41": printed(m["max"], p41),
                "published_table42": published_value(p42),
                "ours_as_printed_table42": printed(m["max"], p42),
                "published_iterations": (
                    T42_ITERATIONS[name][i]
                    if scheme == "filtered" and stencil == 17 and name in T42_ITERATIONS
                    else ""
                ),
                "iterations": m["iterations"],
                "converged": m["converged"],
                "residual_max": m.get("residual_max", ""),
                f"max_err_{BACK}_steps_before": back,
                "monotone_fraction": m.get("monotone_fraction", ""),
                "filter_width": m.get("filter_width", ""),
                "nonconvex_fraction": m["nonconvex_fraction"],
                "time_s": f"{m['time']:.2f}",
            }
        )
    return rows


@dataclasses.dataclass(frozen=True, kw_only=True)
class UnitSlopeMonotone17(Monotone17):
    """Monotone17 with the slope below zero kappa = 1 at every node.

    The papers' operator sum min{D, delta} below zero has slope 1, where
    Monotone17 takes sqrt(max(f, mean f, 1)), large on T2.
    """

    def discretize(self, problem: Problem) -> MonotoneGrid:
        """Monotone17's grid with kappa = 1."""
        grid = super().discretize(problem)
        return dataclasses.replace(grid, kappa=np.ones_like(grid.kappa))


def checks(task: tuple[str, str, int, Any, bool]) -> dict[str, Any]:
    """One row of filtered_checks.csv.

    ``task`` is (check, problem, N, variant, log), all at n = N - 1 with the
    17-point stencil: the variant is the filter width's name for "width",
    the start for "start", delta for "regularised", unused for "kappa",
    "newton" and "catalogue".
    """
    check, name, N, variant, log = task
    n = N - 1
    problem = problem_for(name, n)
    epsilon: Any = ""
    if check == "kappa":
        scheme = Filtered(
            accurate=Standard9(n=n), monotone=UnitSlopeMonotone17(n=n, stencil=17)
        )
        solver: Solver = Newton(scheme=scheme, max_iter=200)
    elif check == "width":
        # Oberman's angular resolution is half the gap between directions;
        # the default dtheta is the full gap.
        half_gap = math.sqrt(1.0 / n) + angular_resolution(17) / 10
        full_gap = math.sqrt(1.0 / n) + 2 * angular_resolution(17) / 10
        epsilon = {
            "half_gap": half_gap,
            "default": full_gap,
            "twice": 2 * full_gap,
            "four_times": 4 * full_gap,
        }[variant]
        solver = FDFiltered(n=n, epsilon=epsilon)
    elif check == "newton":
        solver = FDNewton(n=n)
    elif check == "start":
        start = FDMonotone(n=n) if variant == "monotone" else FDPoisson(n=n)
        scheme = Filtered(accurate=Standard9(n=n), monotone=Monotone17(n=n))
        solver = Newton(scheme=scheme, start=start, max_iter=200)
    elif check == "regularised":
        monotone = SmoothMonotone17(n=n, delta=variant)
        scheme = Filtered(accurate=Standard9(n=n), monotone=monotone)
        solver = Newton(scheme=scheme, max_iter=200)
    else:  # "catalogue"
        solver = FDFiltered(n=n)
    run_name = f"check={check} N={N} variant={variant}"
    m, _ = solve(solver, problem, run_name, log)
    m.setdefault("max", "")  # T7 has no exact solution
    print(f"{name} {run_name}: max {m['max']} its {m['iterations']}", flush=True)
    return {
        "check": check,
        "problem": name,
        "N": N,
        "n_cells": n,
        "variant": variant,
        "epsilon": epsilon,
        "max_err_nodes": m["max"],
        "iterations": m["iterations"],
        "converged": m["converged"],
        "residual_max": m["residual_max"],
        "monotone_fraction": m.get("monotone_fraction", ""),
        "filter_width": m.get("filter_width", ""),
        "nonconvex_fraction": m["nonconvex_fraction"],
        "time_s": f"{m['time']:.2f}",
    }


CHECK_TASKS: list[tuple[str, str, int, Any]] = [
    *(("kappa", "T2", N, "") for N in (63, 127, 255)),
    *(
        ("width", name, N, width)
        for name in ("T20", "T2")
        for N in (63, 127)
        for width in ("half_gap", "default", "twice", "four_times")
    ),
    *(
        ("start", name, N, start)
        for name in ("T20", "T2")
        for N in (63, 127)
        for start in ("monotone", "standard")
    ),
    *(("newton", "T20", N, "") for N in (63, 127)),
    *(
        ("regularised", name, N, delta)
        for name in ("T2", "T1", "T20")
        for N in (63, 127)
        for delta in (1e-4, 1e-2, 1e-1)
    ),
    *(("catalogue", name, N, "") for name in CATALOGUE for N in (17, 65)),
]
"""(check, problem, N, variant) of every row of filtered_checks.csv."""


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
    parser.add_argument("--processes", type=int, default=16, help="worker processes")
    parser.add_argument(
        "--checks", action="store_true", help="write filtered_checks.csv only"
    )
    args = parser.parse_args()
    log = not args.no_mlflow
    if args.checks:
        with Pool(args.processes) as pool:
            rows = pool.map(checks, [(*task, log) for task in CHECK_TASKS], 1)
        write(HERE / "filtered_checks.csv", rows)
        return
    tasks = [
        (name, scheme, stencil, boundary, convention, log)
        for name in ("C2-centred", "T20", "T2", "T21")
        for scheme in ("filtered", "monotone")
        for stencil in STENCILS
        for boundary in ("narrow", "interpolate")
        for convention in ("1/(N-1)", "1/N")
        if not (stencil == 9 and boundary == "interpolate")  # the same stencil
        if convention == "1/(N-1)" or (boundary == "narrow" and stencil == 17)
    ]
    tasks += [
        (name, "standard", 9, "", convention, log)
        for name in ("C2-centred", "T20", "T2", "T21")
        for convention in ("1/(N-1)", "1/N")
    ]
    tasks += [("T1", "filtered", s, "narrow", "1/N", log) for s in STENCILS]
    # The slowest tasks first, so the pool stays busy to the end.
    tasks.sort(key=lambda task: (-task[2], task[4] != "1/(N-1)"))
    with Pool(args.processes) as pool:
        rows = [row for chunk in pool.imap_unordered(sequence, tasks) for row in chunk]
    # Written only after every solve, so no run sees a dirty checkout.
    order = {name: i for i, name in enumerate(("C2-centred", "T20", "T2", "T21", "T1"))}
    rows.sort(
        key=lambda r: (
            order[r["problem"]],
            r["h_convention"],
            r["scheme"],
            r["boundary"],
            r["stencil"],
            r["N"],
        )
    )
    write(HERE / "filtered.csv", rows)


if __name__ == "__main__":
    main()
