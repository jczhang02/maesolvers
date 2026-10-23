"""Regenerate monotone17.csv and monotone17_iterates.csv, or monotone17_checks.csv.

Runs FDMonotone on T1, T2 and T20 over the grid sizes of Froese-Oberman,
J. Comput. Phys. 230 (2011), Table 3 (the SIAM J. Numer. Anal. 49 (2011)
Table 3 sizes are a subset), with the 9-, 17- and 33-point stencils and both
boundary treatments, the 17-point stencil also with h = 1/(N - 1), and on the
cone T21, which the catalogue lacks and this script builds for each grid.
Then it runs Newton from the papers' start for 1 to 8 steps on the coarse
grids (monotone17.md, Finding 2).

With --checks it writes monotone17_checks.csv instead, the evidence for the
design choices and the T20 finding: Newton on the bare positive-part scheme
(kappa = 0, the papers' floored Jacobian) from two starts; Newton on the
papers' regularised scheme MA^delta for several delta; the standard scheme
under the Poisson iteration on T20; and the papers' Jacobian floor at
N = 181.

Every solve is one MLflow run in maesolvers-monotone17-<problem> unless
--no-mlflow is given. The solves run in --processes worker processes
(default 12), so set one thread each. Run it on a100server from the
repository root::

    just remote-bg monotone17 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_monotone17.py
    just remote-bg monotone17-checks env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_monotone17.py --checks --processes 4
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from maesolvers import (
    T1,
    T2,
    T20,
    Box,
    Dirichlet,
    FDMonotone,
    FDPoisson,
    Monotone17,
    Newton,
    Problem,
    Result,
    Solver,
    tracking,
)
from maesolvers.classical.monotone import MonotoneGrid
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


def _jacobian(grid: MonotoneGrid, weights: np.ndarray) -> sp.csc_matrix:
    """``sum over pairs p and directions k of diag(weights[p, k]) D_pk``."""
    weights = weights.ravel()
    m = grid.f.size
    rows = np.tile(np.arange(m), 2 * grid.frames)
    collapse = sp.csr_matrix(
        (weights, (rows, np.arange(weights.size))), shape=(m, weights.size)
    )
    return sp.csc_matrix(collapse @ grid.operators)


@dataclass(frozen=True, kw_only=True)
class BareMonotone17(Monotone17):
    """Froese and Oberman's scheme as printed: min over pairs of a^+ b^+ = f.

    No slope below zero (kappa = 0), and their Jacobian (SINUM Section 5):
    Danskin's theorem on the active pair, each factor max(D u, floor) whatever
    its sign, with their floor 1e-8 / (2 h^2) (Section 5.3).
    """

    jacobian_floor: float = 1e-8

    def residual(self, grid: MonotoneGrid, v: np.ndarray) -> np.ndarray:
        """``min over available pairs of a^+ b^+ - f``."""
        return self._bare(grid, v).min(axis=0) - grid.f

    def jacobian(self, grid: MonotoneGrid, v: np.ndarray) -> sp.csc_matrix:
        """The papers' floored Danskin Jacobian."""
        d = self.directional_differences(grid, v)
        chosen = np.arange(grid.frames)[:, None] == self._bare(grid, v).argmin(axis=0)
        weights = np.maximum(d[:, ::-1], grid.floor) * chosen[:, None, :]
        return _jacobian(grid, weights)

    def _bare(self, grid: MonotoneGrid, v: np.ndarray) -> np.ndarray:
        d = self.directional_differences(grid, v)
        values = np.maximum(d[:, 0], 0.0) * np.maximum(d[:, 1], 0.0)
        return np.where(grid.available, values, np.inf)


def _smooth_max0(a: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """max^delta(a, 0) = (a + sqrt(a^2 + delta^2)) / 2 and its derivative, stably."""
    s = np.hypot(a, delta)
    gap = delta**2 / (2.0 * (s + np.abs(a)))  # (s - |a|) / 2
    slope = gap / s  # (1 - |a| / s) / 2
    return np.maximum(a, 0.0) + gap, np.where(a >= 0.0, 1.0 - slope, slope)


def _smooth_min(
    x: np.ndarray, y: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """min^delta(x, y) = (x + y - sqrt((x - y)^2 + delta^2)) / 2 and its partials."""
    t = x - y
    s = np.hypot(t, delta)
    gap = delta**2 / (2.0 * (s + np.abs(t)))
    slope = gap / s
    dx = np.where(t <= 0.0, 1.0 - slope, slope)
    return np.minimum(x, y) - gap, dx, 1.0 - dx


@dataclass(frozen=True, kw_only=True)
class SmoothMonotone17(Monotone17):
    """The regularised scheme (MA)^delta of SINUM Section 3.5.

    Each positive part is max^delta(D u, 0) and the minimum over the pairs a
    succession of min^delta, in the order of stencil_frames, over the pairs
    available at the node. Smooth, so its Jacobian is the exact one.
    """

    delta: float = 1e-2

    def residual(self, grid: MonotoneGrid, v: np.ndarray) -> np.ndarray:
        """``MA^delta[u] - f``."""
        return self._value_and_weights(grid, v)[0] - grid.f

    def jacobian(self, grid: MonotoneGrid, v: np.ndarray) -> sp.csc_matrix:
        """The Jacobian of :meth:`residual`."""
        return _jacobian(grid, self._value_and_weights(grid, v)[1])

    def _value_and_weights(
        self, grid: MonotoneGrid, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        d = self.directional_differences(grid, v)
        plus, dplus = _smooth_max0(d, self.delta)
        products = plus[:, 0] * plus[:, 1]
        # d products[p] / d d[p, k]
        local = dplus * plus[:, ::-1]
        # value, and its derivative in each pair's product, built pair by pair
        value = np.full(grid.f.size, np.nan)
        dvalue = np.zeros((grid.frames, grid.f.size))
        for p in range(grid.frames):
            use = grid.available[p]
            first = use & np.isnan(value)
            value[first] = products[p, first]
            dvalue[p, first] = 1.0
            more = use & ~first
            new, dx, dy = _smooth_min(value[more], products[p, more], self.delta)
            value[more] = new
            dvalue[:, more] *= dx
            dvalue[p, more] = dy
        return value, dvalue[:, None, :] * local


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[dict[str, Any], Result]:
    """Solve and log to MLflow if ``log``; ``evaluate`` with the info, and the result."""
    if not log:
        result = solver.solve(problem)
        return {**evaluate(result), **result.info}, result
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-monotone17-{problem.name}", run_name=run_name):
        result = solver.solve(problem, callbacks=[logger])
    assert logger.evaluation is not None  # set when the solve ends
    return {**logger.evaluation, **result.info}, result


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
        if stencil == 17:
            # The other grid convention for the JCP column (Finding 3).
            grids += [("1/(N-1)", N, N - 1) for N in JCP_N]
    for convention, N, n in grids:
        problem = cone(n) if name == "T21" else {"T1": T1, "T2": T2, "T20": T20}[name]
        solver = FDMonotone(n=n, stencil=stencil, boundary=boundary)
        run_name = f"N={N} h={convention} stencil={stencil} {boundary}"
        m, _ = solve(solver, problem, run_name, log)
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
            m, _ = solve(solver, problem, run_name, log)
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


def checks(task: tuple[str, str, int, int, Any, bool]) -> dict[str, Any]:
    """One row of monotone17_checks.csv.

    ``task`` is (check, problem, N, stencil, variant, log); the variant is the
    start for "bare", delta for "delta", the floor for "floor", unused else.
    """
    check, name, N, stencil, variant, log = task
    problem = {"T1": T1, "T2": T2, "T20": T20}[name]
    start, delta, floor = "", "", ""
    if check == "bare":
        start = variant
        start_class = PapersStartNewton if variant == "papers" else Newton
        solver: Solver = start_class(
            scheme=BareMonotone17(n=N, stencil=stencil), max_iter=200
        )
        floor = 1e-8
    elif check == "delta":
        delta = variant
        solver = Newton(
            scheme=SmoothMonotone17(n=N, stencil=stencil, delta=variant), max_iter=200
        )
    elif check == "poisson":
        solver = FDPoisson(n=N)
    else:  # "floor"
        floor = variant
        solver = Newton(
            scheme=Monotone17(n=N, stencil=stencil, jacobian_floor=variant),
            max_iter=1000,
        )
    run_name = f"check={check} N={N} stencil={stencil} variant={variant}"
    m, result = solve(solver, problem, run_name, log)
    print(f"{name} {run_name}: max {m['max']:.4g} its {m['iterations']}", flush=True)
    min_difference: Any = ""
    if check != "poisson":
        scheme: Any = solver.scheme  # type: ignore[attr-defined]
        grid = scheme.discretize(problem)
        d = scheme.directional_differences(grid, result.values[grid.interior])
        min_difference = float(
            d[np.broadcast_to(grid.available[:, None], d.shape)].min()
        )
    return {
        "check": check,
        "problem": name,
        "N": N,
        "stencil": stencil if check != "poisson" else 9,
        "start": start,
        "delta": delta,
        "jacobian_floor": floor,
        "max_err_nodes": m["max"],
        "iterations": m["iterations"],
        "converged": m["converged"],
        "residual_max": m["residual_max"],
        "min_available_difference": min_difference,
        "nonconvex_fraction": m["nonconvex_fraction"],
        "time_s": f"{m['time']:.2f}",
    }


CHECK_TASKS: list[tuple[str, str, int, int, Any]] = [
    *(("floor", "T20", 181, 17, floor) for floor in (1e-8, 1e-12)),  # slowest first
    *(
        ("bare", name, N, stencil, start)
        for name, N, stencil in (
            ("T20", 31, 9),
            ("T20", 31, 17),
            ("T20", 31, 33),
            ("T20", 63, 17),
            ("T2", 31, 17),
            ("T1", 31, 17),
        )
        for start in ("ours", "papers")
    ),
    *(
        ("delta", name, N, 17, delta)
        for name, N in (("T20", 31), ("T20", 63), ("T1", 31))
        for delta in (1.0, 1e-1, 1e-2, 1e-3, 1e-4)
    ),
    *(("poisson", "T20", N, 9, "") for N in (31, 45, 63)),
]
"""(check, problem, N, stencil, variant) of every row of monotone17_checks.csv."""


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV, the header from the first row."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse the command line, run the solves, write both CSVs.

    The sequences run in parallel; the iterates, many short runs, run one
    after another, since a burst of short runs from many processes made the
    shared MLflow server's SQLite store fail an insert.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=12, help="worker processes")
    parser.add_argument(
        "--checks", action="store_true", help="write monotone17_checks.csv only"
    )
    args = parser.parse_args()
    log = not args.no_mlflow
    if args.checks:
        with Pool(args.processes) as pool:
            rows = pool.map(checks, [(*task, log) for task in CHECK_TASKS], 1)
        write(HERE / "monotone17_checks.csv", rows)
        return
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
        rows = [row for chunk in pool.imap(sequence, main_tasks) for row in chunk]
    iterate_rows = [iterates(task) for task in iterate_tasks]
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
