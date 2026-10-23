"""Regenerate multigrid.csv: the tables of Liu, Froese, Oberman and Xiao (2017).

J. Liu, B. D. Froese, A. M. Oberman, M. Xiao, "A multigrid scheme for 3D
Monge-Ampere equations", Int. J. Comput. Math. 94 (2017) 1850-1866: Tables 1
and 2 (Examples 4.1 to 4.3, T1, T3 and T2 on (0, 1)^2, N = 128 to 8192,
red-black and lexicographic Gauss-Seidel), Table 3 (Example 4.4, T9 on
(0, 1)^3), Table 5 (Example 4.5, T46 on (0, pi)^3), Table 6 (Example 4.6, T28)
and Table 7 (Example 4.7, T22), N = 8 to 128, lexicographic. The published
values below were checked against the page images.

One row per table, ordering and N. The FMG-FAS columns are
``FDMultigrid(n=N, ordering=...)`` with its defaults (the paper's: 2 + 2
sweeps, 1 on the coarsest grid, relative residual 1e-6); ``error`` is the
max error at the nodes, as ``evaluate`` reports it. ``error_converged`` is
the error of the discrete solution itself, the same solver with
``tol=1e-12``. In 3D a red-black row is ours alone (the paper sweeps
lexicographically there). For N <= 32 in 3D the Gauss-Seidel columns
repeat the paper's stand-alone solver: lexicographic sweeps from
``1.01 u`` at the unknowns until the residual's discrete L2 norm is 1e-6
times that of the start.

Every row is one MLflow run in maesolvers-multigrid-<problem>, named
``table<k>-<ordering>-n<N>``, unless --no-mlflow is given. The cases run in
--workers processes (default 8), one thread each, the largest first; the
CSV is written after the last run. Run it on a100server from the
repository root, writing outside the checkout so the runs see it clean::

    just remote-bg multigrid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benchmarks/run_multigrid.py \\
        --out /jczhang02/maesolvers-data/data/multigrid.csv

``--report`` prints the markdown tables of ``benchmarks/multigrid.md`` from
the CSV instead: ours next to the published values, the ratios and the
least-squares orders.
"""

from __future__ import annotations

import argparse
import csv
import platform
import resource
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from maesolvers import CATALOGUE, FDMultigrid, Problem, StandardBox, tracking
from maesolvers.classical.multigrid import sweep_groups
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
CSV = HERE / "multigrid.csv"

GS_TOL = 1e-6
"""The stand-alone Gauss-Seidel solver's relative residual (the paper's 1e-6)."""
GS_MAX_SWEEPS = 5000
GS_LARGEST_N = 32
"""The paper runs its stand-alone Gauss-Seidel solver up to N = 32."""

TWO_D = (128, 256, 512, 1024, 2048, 4096, 8192)
THREE_D = (8, 16, 32, 64, 128)

# Published values as printed, by (table, problem): one tuple per N of
# (relative residual, error, iterations, CPU seconds); None where not printed.
P = None
PUBLISHED: dict[tuple[int, str], dict[int, tuple[Any, ...]]] = {
    (1, "T1"): dict(zip(TWO_D, [
        (P, "4.5e-06", 1, "0.1"), (P, "1.1e-06", 1, "0.1"), (P, "2.8e-07", 1, "0.3"),
        (P, "7.0e-08", 1, "1.0"), (P, "1.7e-08", 1, "3.6"), (P, "4.3e-09", 1, "16.2"),
        (P, "1.0e-09", 1, "67.1"),
    ], strict=True)),
    (1, "T3"): dict(zip(TWO_D, [
        (P, "3.8e-05", 1, "0.1"), (P, "1.3e-05", 1, "0.1"), (P, "4.7e-06", 1, "0.3"),
        (P, "1.7e-06", 1, "1.1"), (P, "5.9e-07", 1, "4.0"), (P, "2.1e-07", 1, "16.5"),
        (P, "7.4e-08", 1, "69.1"),
    ], strict=True)),
    (1, "T2"): dict(zip(TWO_D, [
        (P, "8.9e-03", 2, "0.1"), (P, "6.4e-03", 2, "0.2"), (P, "4.5e-03", 1, "0.3"),
        (P, "3.2e-03", 1, "1.1"), (P, "2.3e-03", 1, "3.8"), (P, "1.6e-03", 1, "18.0"),
        (P, "1.1e-03", 1, "70.3"),
    ], strict=True)),
    (2, "T1"): dict(zip(TWO_D, [
        (P, "5.1e-06", 1, "0.1"), (P, "1.3e-06", 1, "0.2"), (P, "3.3e-07", 1, "0.8"),
        (P, "8.2e-08", 1, "3.2"), (P, "2.1e-08", 1, "12.7"), (P, "5.1e-09", 1, "53.2"),
        (P, "1.3e-09", 1, "245.6"),
    ], strict=True)),
    (2, "T3"): dict(zip(TWO_D, [
        (P, "4.3e-05", 1, "0.1"), (P, "1.5e-05", 1, "0.3"), (P, "5.4e-06", 1, "0.9"),
        (P, "1.9e-06", 1, "3.27"), (P, "6.8e-07", 1, "12.9"), (P, "2.4e-07", 1, "54.6"),
        (P, "8.5e-08", 1, "259.4"),
    ], strict=True)),
    (2, "T2"): dict(zip(TWO_D, [
        (P, "8.9e-03", 3, "0.1"), (P, "6.3e-03", 2, "0.2"), (P, "4.4e-03", 1, "0.8"),
        (P, "3.2e-03", 1, "3.1"), (P, "2.2e-03", 1, "12.6"), (P, "1.6e-03", 1, "53.5"),
        (P, "1.1e-03", 1, "250.7"),
    ], strict=True)),
    (3, "T9"): dict(zip(THREE_D, [
        ("3.1e-07", "1.2e-03", 1, "0.1"), ("9.5e-09", "2.7e-04", 1, "0.2"),
        ("2.1e-10", "5.7e-05", 1, "2.0"), ("4.1e-12", "1.4e-05", 1, "16.9"),
        ("5.2e-13", "6.2e-06", 1, "136.9"),
    ], strict=True)),
    (5, "T46"): dict(zip(THREE_D, [
        ("6.8e-08", "1.7e-02", 2, "0.1"), ("2.1e-08", "3.3e-03", 1, "0.2"),
        ("2.7e-10", "8.1e-04", 1, "2.0"), ("3.7e-12", "2.1e-04", 1, "16.4"),
        ("4.8e-13", "5.2e-05", 1, "134.1"),
    ], strict=True)),
    (6, "T28"): dict(zip(THREE_D, [
        ("2.7e-07", "5.3e-04", 1, "0.1"), ("1.1e-08", "2.1e-04", 1, "0.2"),
        ("4.8e-10", "7.4e-05", 1, "2.0"), ("2.0e-11", "2.6e-05", 1, "16.4"),
        ("1.2e-12", "9.3e-06", 1, "133.9"),
    ], strict=True)),
    (7, "T22"): dict(zip(THREE_D, [
        ("1.7e-07", "1.4e-03", 1, "0.1"), ("2.1e-08", "1.4e-03", 1, "0.2"),
        ("2.7e-08", "4.3e-03", 1, "2.0"), ("4.9e-09", "2.2e-03", 1, "16.3"),
        ("1.7e-09", "1.6e-03", 1, "133.1"),
    ], strict=True)),
}  # fmt: skip

PUBLISHED_GS: dict[int, dict[int, tuple[str, str, int, str]]] = {
    3: {8: ("9.3e-07", "1.2e-03", 72, "0.4"), 16: ("9.9e-07", "3.0e-04", 270, "11.5"),
        32: ("1.0e-06", "8.1e-05", 867, "324.6")},
    5: {8: ("9.2e-07", "1.7e-02", 82, "0.4"), 16: ("9.8e-07", "4.3e-03", 296, "12.9"),
        32: ("1.0e-06", "1.1e-03", 954, "356.2")},
    6: {8: ("9.7e-07", "4.0e-04", 86, "0.4"), 16: ("9.9e-07", "1.4e-04", 312, "13.2"),
        32: ("9.9e-07", "5.1e-05", 925, "345.8")},
    7: {8: ("8.9e-07", "1.4e-03", 75, "0.4"), 16: ("9.7e-07", "1.4e-03", 266, "12.2"),
        32: ("9.9e-07", "1.1e-03", 854, "320.2")},
}  # fmt: skip
"""The stand-alone Gauss-Seidel columns of Tables 3 and 5 to 7, by table and N."""

EXAMPLE = {"T1": "4.1", "T3": "4.2", "T2": "4.3", "T9": "4.4", "T46": "4.5"}
EXAMPLE |= {"T28": "4.6", "T22": "4.7"}

FIELDS = (
    "table",
    "example",
    "problem",
    "ordering",
    "n",
    "relres",
    "error",
    "iterations",
    "time_s",
    "published_relres",
    "published_error",
    "published_iterations",
    "published_time_s",
    "error_after_fmg",
    "error_converged",
    "iterations_converged",
    "gs_relres",
    "gs_error",
    "gs_iterations",
    "gs_time_s",
    "published_gs_relres",
    "published_gs_error",
    "published_gs_iterations",
    "published_gs_time_s",
    "max_grid",
    "nonconvex_fraction",
    "residual_max",
    "peak_rss_mb",
    "commit",
    "cpu",
)


def cases() -> list[tuple[int, str, str, int]]:
    """Every (table, problem, ordering, N), the largest grids first."""
    out = []
    for (table, problem), rows in PUBLISHED.items():
        orderings = (
            ("red-black",)
            if table == 1
            else ("lexicographic",)
            if table == 2
            else ("lexicographic", "red-black")
        )
        out += [(table, problem, o, n) for o in orderings for n in rows]
    return sorted(out, key=lambda c: -(c[3] ** CATALOGUE[c[1]].dim))


def gauss_seidel(problem: Problem, n: int) -> dict[str, float]:
    """The paper's stand-alone solver: lexicographic sweeps from 1.01 u."""
    assert problem.exact is not None
    scheme = StandardBox(n=n)
    grid = scheme.discretize(problem)
    box = grid.grid
    u = grid.values(1.01 * problem.exact(box.nodes[box.interior]))
    f = np.zeros(len(u))
    f[box.interior] = grid.f
    groups = sweep_groups(n, box.dim, "lexicographic")

    def residual() -> float:
        return float(np.linalg.norm(scheme.residual(grid, u[box.interior])))

    start, reference, sweeps = time.perf_counter(), residual(), 0
    relres = 1.0
    while relres > GS_TOL and sweeps < GS_MAX_SWEEPS:
        for nodes in groups:
            u[nodes] = scheme.relax(grid, u, f[nodes], nodes)
        sweeps += 1
        relres = residual() / reference
    return {
        "gs_relres": relres,
        "gs_error": float(np.max(np.abs(u - problem.exact(box.nodes)))),
        "gs_iterations": sweeps,
        "gs_time_s": time.perf_counter() - start,
    }


def run_case(case: tuple[int, str, str, int], log: bool) -> dict[str, Any]:
    """Run one row of multigrid.csv."""
    table, name, ordering, n = case
    problem = CATALOGUE[name]
    solver = FDMultigrid(n=n, ordering=ordering)
    published = PUBLISHED[table, name][n] if ordering == _paper_ordering(table) else ()
    row: dict[str, Any] = {
        "table": table,
        "example": EXAMPLE[name],
        "problem": name,
        "ordering": ordering,
        "n": n,
        "commit": tracking.git_tags()["mlflow.source.git.commit"],
        "cpu": platform.processor() or platform.machine(),
    }
    for key, value in zip(
        (
            "published_relres",
            "published_error",
            "published_iterations",
            "published_time_s",
        ),
        published,
        strict=False,
    ):
        row[key] = "" if value is None else value

    def solve() -> None:
        callbacks = [tracking.MLflowLogger()] if log else []
        result = solver.solve(problem, callbacks=callbacks)
        metrics = evaluate(result)
        row.update(
            relres=result.info["history"][-1],
            error=metrics["max"],
            iterations=result.info["iterations"],
            time_s=result.info["time"],
            error_after_fmg=metrics["max"],
            max_grid=metrics["max_grid"],
            nonconvex_fraction=metrics["nonconvex_fraction"],
            residual_max=result.info["residual_max"],
        )
        del result
        if row["iterations"] > 1:
            fmg = FDMultigrid(n=n, ordering=ordering, max_iter=1).solve(problem)
            assert fmg.values is not None and fmg.nodes is not None
            row["error_after_fmg"] = _error(problem, fmg.values, fmg.nodes)
            del fmg
        tight = FDMultigrid(n=n, ordering="red-black", tol=1e-12).solve(problem)
        assert tight.values is not None and tight.nodes is not None
        row["error_converged"] = _error(problem, tight.values, tight.nodes)
        row["iterations_converged"] = tight.info["iterations"]
        del tight
        if problem.dim == 3 and ordering == "lexicographic" and n <= GS_LARGEST_N:
            row.update(gauss_seidel(problem, n))
            for key, value in zip(
                ("relres", "error", "iterations", "time_s"),
                PUBLISHED_GS[table][n],
                strict=True,
            ):
                row[f"published_gs_{key}"] = value
        row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    if log:
        import mlflow

        experiment = f"maesolvers-multigrid-{name}"
        params = {"table": table, "example": EXAMPLE[name]}
        with tracking.run(experiment, params, run_name=f"table{table}-{ordering}-n{n}"):
            solve()
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in row.items()
                    if k.startswith(("error_", "gs_", "iterations_")) and v != ""
                }
            )
    else:
        solve()
    print(", ".join(f"{k} {row.get(k, '')}" for k in FIELDS[:9]), flush=True)
    return row


def _error(problem: Problem, values: np.ndarray, nodes: np.ndarray) -> float:
    """The max error at the nodes, evaluate's ``max``."""
    assert problem.exact is not None
    return float(np.max(np.abs(values - problem.exact(nodes))))


def _paper_ordering(table: int) -> str:
    return "red-black" if table == 1 else "lexicographic"


def _work(args: tuple[tuple[int, str, str, int], bool]) -> dict[str, Any]:
    return run_case(*args)


def main(argv: Iterable[str] | None = None) -> None:
    """Parse the command line; run the cases, or print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--largest", type=int, default=None, help="skip larger N")
    parser.add_argument("--out", type=Path, default=CSV)
    parser.add_argument("--report", action="store_true", help="print the md tables")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.report:
        report(args.out)
        return
    todo = [c for c in cases() if args.largest is None or c[3] <= args.largest]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(_work, [(c, not args.no_mlflow) for c in todo]))
    rows.sort(key=lambda r: (r["table"], r["problem"], r["ordering"], r["n"]))
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(path: Path) -> None:
    """Print, per table and problem, ours next to the published values."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for (table, name), _ in PUBLISHED.items():
        for ordering in ("red-black", "lexicographic"):
            sel = [
                r
                for r in rows
                if r["table"] == str(table)
                and r["problem"] == name
                and r["ordering"] == ordering
            ]
            if not sel:
                continue
            sel.sort(key=lambda r: int(r["n"]))
            print(f"\nTable {table}, {name} (Example {EXAMPLE[name]}), {ordering}\n")
            print(
                "| N | Published | Ours | Ratio | Iter (pub., ours) | RelRes (pub., ours)"
                " | Converged | CPU s (pub., ours) |"
            )
            print("|---|---|---|---|---|---|---|---|")
            ns, ours, pub = [], [], []
            for r in sel:
                ratio = (
                    f"{float(r['error']) / float(r['published_error']):.2f}"
                    if r["published_error"]
                    else ""
                )
                print(
                    f"| {r['n']} | {r['published_error']} | {float(r['error']):.3e} | {ratio}"
                    f" | {r['published_iterations']}, {r['iterations']}"
                    f" | {r['published_relres']}, {float(r['relres']):.1e}"
                    f" | {float(r['error_converged']):.3e}"
                    f" | {r['published_time_s']}, {float(r['time_s']):.1f} |"
                )
                ns.append(int(r["n"]))
                ours.append(float(r["error"]))
                if r["published_error"]:
                    pub.append(float(r["published_error"]))
            if len(ns) < 2:
                continue
            print(f"\norder ours {_order(ns, ours):.2f}", end="")
            if len(pub) == len(ns):
                print(f", published {_order(ns, pub):.2f}", end="")
            print()
            gs = [r for r in sel if r["gs_error"]]
            if gs:
                print("\n| N | GS published | GS ours | Sweeps (pub., ours) |")
                print("|---|---|---|---|")
                for r in gs:
                    print(
                        f"| {r['n']} | {r['published_gs_error']} | {float(r['gs_error']):.3e}"
                        f" | {r['published_gs_iterations']}, {r['gs_iterations']} |"
                    )


def _order(ns: list[int], errors: list[float]) -> float:
    """The least-squares slope of -log(error) against log N."""
    return float(-np.polyfit(np.log(ns), np.log(errors), 1)[0])


if __name__ == "__main__":
    main()
