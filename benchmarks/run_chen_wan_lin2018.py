"""Regenerate chen_wan_lin2018.csv, chen_wan_lin2018_table5.csv or the checks.

Runs the examples of Chen-Wan-Lin, J. Sci. Comput. 76 (2018), Section 7, at
the paper's N = 32, 64, ..., 512 with n = N cells per side (Finding 1 of
chen_wan_lin2018.md): Tables 1 to 3, the mixed scheme (FDMixed) and the pure
semi-Lagrangian scheme (PolicyIteration on MixedHJB with wide_search="only")
on T1 over [-1, 1]^2, T2 and T20c; Table 4, the mixed scheme on the cone T21
over (-0.5, 0.5)^2; with --table5, Table 5, u_h(0, 0) of the mixed scheme
under either search rule of the wide stencil ("mixed", the default, and
"everywhere") and of the standard 9-point scheme on f = 1, g = 0 over
(-0.5, 0.5)^2, into chen_wan_lin2018_table5.csv. T1 over [-1, 1]^2, the
cone and the Table 5 problem are built here: the catalogue's T1 is on
(0, 1)^2, and the catalogue cannot hold T21's Dirac mass.

The errors are the paper's: the max over the nodes, and the discrete L2 norm
sqrt(h^2 sum e^2) over the nodes, which matches its tables (Finding 1).

With --checks it writes chen_wan_lin2018_checks.csv instead, the evidence for
the findings: the grid convention (n = N against n = N + 1), the search of
the wide-stencil controls at every node, the discrete Dirac mass of the cone,
the number of angles, and policy iteration from the exact solution. With
--cycling it writes chen_wan_lin2018_cycling.csv, the evidence on the default
rule's cycling (Findings 6 and 7): the default rule from the root of
wide_search="everywhere" on Table 5's problem at N = 256, and both rules on
the catalogue's T4 at n = 16 to 128.

Every solve is one MLflow run in maesolvers-mixed-<problem> unless
--no-mlflow is given. The solves run in --processes worker processes
(default 16), the longest first, so set one thread each. Run it on
a100server from the repository root::

    just remote-bg cwl env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_chen_wan_lin2018.py
    just remote-bg cwl-checks env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_chen_wan_lin2018.py --checks
    just remote-bg cwl-table5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_chen_wan_lin2018.py --table5
    just remote-bg cwl-cycling env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/run_chen_wan_lin2018.py --cycling
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from run_monotone17 import printed, published_value, write

from maesolvers import (
    T1,
    T2,
    T4,
    Box,
    Dirichlet,
    FDMixed,
    FDPoisson,
    MixedHJB,
    PolicyIteration,
    Problem,
    Result,
    Solver,
    T20c,
    tracking,
)
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent
NS = (32, 64, 128, 256, 512)
HALF = Box((-0.5, -0.5), (0.5, 0.5))
KRYLOV_TOL = 1e-12
POISSON_MAX_ITER = 300_000
"""Steps of the Poisson iteration of Table 5's standard column; the default
10 000 stops it unconverged from N = 128 on (chen_wan_lin2018.md, Table 5)."""
SEVEN_POINT_CONE_MASS = 4.0 * math.sqrt(2.0) - 2.0
"""The mass at which the exact cone solves the 7-point equation at its tip (Finding 3)."""

T1_SQUARE = Problem(
    "T1-square",
    Box((-1.0, -1.0), (1.0, 1.0)),
    T1.f,
    T1.boundary,
    exact=T1.exact,
    note="T1 on [-1, 1]^2 (Chen-Wan-Lin 2018, Example 1).",
)


def _zero(x: np.ndarray) -> np.ndarray:
    return np.zeros(np.shape(x)[:-1])


def _one(x: np.ndarray) -> np.ndarray:
    return np.ones(np.shape(x)[:-1])


def _cone_exact(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(x, dtype=float) ** 2, axis=-1))


def cone(n: int, mass: float = SEVEN_POINT_CONE_MASS) -> Problem:
    """T21 on (-0.5, 0.5)^2 for the grid of n cells: f = mass / h^2 at the origin.

    The paper's f = pi delta, with n even so that the origin is a node; the
    paper does not say how it discretises the Dirac mass (Finding 3).
    """
    h = 1.0 / n

    def f(x: np.ndarray) -> np.ndarray:
        tip = np.all(np.abs(np.asarray(x, dtype=float)) < 0.25 * h, axis=-1)
        return np.where(tip, mass / h**2, 0.0)

    return Problem(
        "T21",
        HALF,
        f,
        Dirichlet(_cone_exact),
        exact=_cone_exact,
        note=f"The cone |x| on (-0.5, 0.5)^2, f = {mass:.6g} / h^2 at the origin.",
    )


T7_HALF = Problem(
    "T7-half",
    HALF,
    _one,
    Dirichlet(_zero),
    observables={"u_origin": lambda u: float(u(np.zeros(2)))},
    note="f = 1, g = 0 on (-0.5, 0.5)^2 (Chen-Wan-Lin 2018, Example 5): T7 "
    "under x -> 2x, u -> (u - 1)/4.",
)

PROBLEMS = {
    "T1-square": T1_SQUARE,
    "T2": T2,
    "T4": T4,
    "T20c": T20c,
    "T7-half": T7_HALF,
}


def _sci(*values: str, power: int) -> tuple[tuple[str, int], ...]:
    return tuple((v, power) for v in values)


# Chen-Wan-Lin 2018, Tables 1 to 4, as printed: (mantissa, power of ten) in
# the order of NS; each table prints the mantissa with four digits.
PUBLISHED: dict[tuple[str, str], dict[str, Any]] = {
    ("T1-square", "mixed"): {
        "table": 1,
        "l2": (
            ("1.201", -3),
            ("3.009", -4),
            ("7.526", -5),
            ("1.882", -5),
            ("4.705", -6),
        ),
        "linf": (
            ("9.598", -4),
            ("2.404", -4),
            ("6.013", -5),
            ("1.504", -5),
            ("3.759", -6),
        ),
        "iterations": (4, 4, 4, 4, 4),
    },
    ("T1-square", "wide only"): {
        "table": 1,
        "l2": (
            ("1.868", -2),
            ("1.020", -2),
            ("5.263", -3),
            ("2.801", -3),
            ("1.600", -3),
        ),
        "linf": (
            ("1.557", -2),
            ("8.364", -3),
            ("4.240", -3),
            ("2.259", -3),
            ("1.268", -3),
        ),
        "iterations": (5, 5, 6, 5, 5),
    },
    ("T2", "mixed"): {
        "table": 2,
        "l2": (
            ("6.450", -5),
            ("1.628", -5),
            ("4.084", -6),
            ("1.022", -6),
            ("2.557", -7),
        ),
        "linf": (
            ("2.359", -4),
            ("8.211", -5),
            ("2.882", -5),
            ("1.015", -5),
            ("3.583", -6),
        ),
        "iterations": (4, 5, 5, 5, 5),
    },
    ("T2", "wide only"): {
        "table": 2,
        "l2": (
            ("1.493", -3),
            ("9.634", -4),
            ("5.166", -4),
            ("3.153", -4),
            ("1.583", -4),
        ),
        "linf": (
            ("5.799", -3),
            ("4.394", -3),
            ("2.697", -3),
            ("1.824", -3),
            ("1.120", -3),
        ),
        "iterations": (5, 4, 5, 5, 5),
    },
    ("T20c", "mixed"): {
        "table": 3,
        "l2": (
            ("1.270", -4),
            ("4.273", -5),
            ("1.835", -5),
            ("1.544", -5),
            ("3.396", -6),
        ),
        "linf": (
            ("4.298", -4),
            ("1.520", -4),
            ("6.907", -5),
            ("5.959", -5),
            ("1.513", -5),
        ),
        "iterations": (4, 6, 7, 9, 20),
    },
    ("T20c", "wide only"): {
        "table": 3,
        "l2": (
            ("1.337", -3),
            ("9.084", -4),
            ("6.940", -4),
            ("3.815", -4),
            ("1.998", -4),
        ),
        "linf": (
            ("6.604", -3),
            ("3.304", -3),
            ("1.901", -3),
            ("9.335", -4),
            ("4.563", -4),
        ),
        "iterations": (5, 6, 7, 7, 9),
    },
    ("T21", "mixed"): {
        "table": 4,
        "l2": (
            ("1.156", -3),
            ("6.484", -4),
            ("3.803", -4),
            ("2.159", -4),
            ("1.148", -4),
        ),
        "linf": (
            ("3.868", -3),
            ("2.583", -3),
            ("1.848", -3),
            ("1.305", -3),
            ("9.203", -4),
        ),
        "iterations": (9, 15, 17, 23, 27),
    },
}

# Table 5: u_h(0, 0) by the monotone mixed scheme and by the non-monotone
# standard scheme, whose solution is concave.
TABLE5 = {
    "mixed": ("-0.18380", "-0.18444", "-0.18461", "-0.18485", "-0.18507"),
    "standard": ("0.18063", "0.18312", "0.18436", "0.18499", "0.18530"),
}
TABLE5_COLUMNS = {"mixed": "mixed", "everywhere": "mixed", "standard": "standard"}
"""Our columns of Table 5 and the published column each is compared with: the
paper does not say which search rule its mixed column used (Finding 2)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExactStart(PolicyIteration):
    """Policy iteration from the exact solution at the unknowns (Finding 5)."""

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """The exact solution at the scheme's unknowns."""
        assert problem.exact is not None
        return problem.exact(context.nodes[context.interior])


@dataclasses.dataclass(frozen=True, kw_only=True)
class RootStart(PolicyIteration):
    """Policy iteration from the root that ``start`` reaches (Finding 6).

    Its first residual, ``Result.info["history"][0]``, is the residual of
    this scheme at that root.
    """

    start: PolicyIteration

    def initial_guess(self, problem: Problem, context: Any) -> np.ndarray:
        """The unknowns at which ``start`` converges."""
        begun = self.start.run(problem, self.start.setup(problem), ())
        assert begun.converged, "the start did not converge"
        return begun.state


WIDE_PROBLEMS = ("T21", "T7-half")
"""Problems on which the mixed scheme takes the wide stencil at many nodes."""


def solver_for(scheme: str, n: int, problem: str, **options: Any) -> Solver:
    """The solver of a column: "mixed", "wide only", "everywhere", "exact start",
    "everywhere start" (the default rule from the root of "everywhere") or
    "standard".

    Policy iteration solves each policy's system by sparse LU, except where
    the wide stencil enters at many nodes ("wide only", "everywhere", T21 and
    T7-half) from n = 256 on: there the LU fills in (tens of GB and hours per
    policy at n = 512), and BiCGSTAB with an incomplete LU solves it to a
    relative residual of 1e-12. Where the LU is cheap it stays, since that
    tolerance moves the fourth digit of the 7-point errors at n = 512. The
    Poisson iteration of the "standard" column gets POISSON_MAX_ITER steps.
    """
    if scheme == "standard":
        return FDPoisson(n=n, max_iter=POISSON_MAX_ITER)
    wide = scheme in ("wide only", "everywhere", "everywhere start")
    wide = wide or problem in WIDE_PROBLEMS
    iteration = {"krylov_tol": KRYLOV_TOL} if wide and n >= 256 else {}
    if scheme == "mixed":
        return FDMixed(n=n, **options, **iteration)
    if scheme == "exact start":
        return ExactStart(scheme=MixedHJB(n=n, **options), **iteration)
    if scheme == "everywhere start":
        start = solver_for("everywhere", n, problem, **options)
        assert isinstance(start, PolicyIteration)
        return RootStart(scheme=MixedHJB(n=n, **options), start=start, **iteration)
    wide_search = "only" if scheme == "wide only" else scheme
    return PolicyIteration(
        scheme=MixedHJB(n=n, wide_search=wide_search, **options), **iteration
    )


def paper_norms(result: Result) -> tuple[float, float]:
    """The paper's L2 norm sqrt(h^2 sum e^2) and the max norm, over the nodes."""
    problem = result.problem
    assert problem.exact is not None and result.nodes is not None
    assert result.values is not None
    error = result.values - problem.exact(result.nodes)
    h = float(result.info["h"][0])
    return float(np.sqrt(h * h * np.sum(error**2))), float(np.max(np.abs(error)))


def origin_value(result: Result) -> float:
    """u_h(0, 0), the value at the node at the origin."""
    assert result.nodes is not None and result.values is not None
    node = np.flatnonzero(np.all(np.abs(result.nodes) < 1e-12, axis=-1))
    assert node.size == 1, "the origin is not a node"
    return float(result.values[node[0]])


def solve(
    solver: Solver, problem: Problem, run_name: str, log: bool
) -> tuple[Result, dict[str, Any]]:
    """Solve, logging to MLflow if ``log``; the result and its tags."""
    if not log:
        return solver.solve(problem), {}
    import mlflow  # the experiments group; only when logging

    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-mixed-{problem.name}", run_name=run_name) as run:
        result = solver.solve(problem, callbacks=[logger])
        if problem.exact is not None:
            l2, linf = paper_norms(result)
            run_metrics = {"paper_l2": l2, "paper_linf": linf}
        else:
            run_metrics = {"u_origin_node": origin_value(result)}
        mlflow.log_metrics(run_metrics)
    return result, {"run_id": run.info.run_id}


def _problem(name: str, n: int, mass: float = SEVEN_POINT_CONE_MASS) -> Problem:
    return cone(n, mass) if name == "T21" else PROBLEMS[name]


def table_row(task: tuple[str, str, int, bool]) -> dict[str, Any]:
    """One row of chen_wan_lin2018.csv or of the Table 5 CSV: a problem, a column, an N."""
    name, scheme, N, log = task
    problem = _problem(name, N)
    solver = solver_for(scheme, N, name)
    result, _ = solve(solver, problem, f"N={N} {scheme}", log)
    info = result.info
    row: dict[str, Any] = {"problem": name, "scheme": scheme, "N": N, "n_cells": N}
    k = NS.index(N)
    if name == "T7-half":
        u0 = origin_value(result)
        reported = -u0 if scheme == "standard" else u0  # the concave root is -u
        published = TABLE5[TABLE5_COLUMNS[scheme]][k]
        row |= {
            "table": 5,
            "u_origin": f"{reported:.6f}",
            "published_u_origin": published,
            "ours_as_printed_u_origin": f"{reported:.5f}",
        }
    else:
        l2, linf = paper_norms(result)
        paper = PUBLISHED[(name, scheme)]
        row |= {
            "table": paper["table"],
            "l2": f"{l2:.6e}",
            "linf": f"{linf:.6e}",
            "published_l2": published_value(paper["l2"][k]),
            "published_linf": published_value(paper["linf"][k]),
            "ours_as_printed_l2": printed(l2, paper["l2"][k]),
            "ours_as_printed_linf": printed(linf, paper["linf"][k]),
            "published_iterations": paper["iterations"][k],
        }
    row |= {
        "iterations": info["iterations"],
        "converged": info["converged"],
        "residual_max": f"{info.get('residual_max', float('nan')):.3e}",
        "wide_fraction": info.get("wide_fraction", ""),
        "angles": info.get("angles", ""),
        # the Poisson iteration takes hundreds of steps: its history stays in MLflow
        "history": ""
        if scheme == "standard"
        else " ".join(f"{r:.3e}" for r in info["history"]),
        "time_s": f"{info['time']:.2f}",
    }
    print(f"{name} {scheme} N={N}: {row.get('linf', row.get('u_origin'))}", flush=True)
    return row


TABLE_TASKS: list[tuple[str, str, int]] = [
    (name, scheme, N) for (name, scheme) in PUBLISHED for N in NS
]
"""(problem, column, N) of every row of chen_wan_lin2018.csv."""

TABLE5_TASKS: list[tuple[str, str, int]] = [
    ("T7-half", scheme, N) for scheme in TABLE5_COLUMNS for N in NS
]
"""(problem, column, N) of every row of chen_wan_lin2018_table5.csv."""


def _order(coarse: str, fine: str) -> float:
    """log2 of the ratio of two errors given as strings, the order on halving h."""
    return math.log2(float(coarse) / float(fine))


def with_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the observed orders log2(e(N/2) / e(N)) of each column, as the paper does."""
    out = []
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["problem"], row["scheme"])
        last = previous.get(key)
        rates = {}
        for norm in ("l2", "linf"):
            if norm not in row:
                continue
            ours = published = ""
            if last is not None:
                ours = f"{_order(last[norm], row[norm]):.2f}"
                published = (
                    f"{_order(last['published_' + norm], row['published_' + norm]):.2f}"
                )
            rates[f"rate_{norm}"] = ours
            rates[f"published_rate_{norm}"] = published
        out.append(row | rates)
        previous[key] = row
    return out


def check_row(task: tuple[str, str, int, dict[str, Any], bool]) -> dict[str, Any]:
    """One row of chen_wan_lin2018_checks.csv.

    ``task`` is (check, problem, N, variant, log). For "grid" the variant
    holds n_cells; for "search" and "start" the column; for "mass" the
    Dirac mass of the cone; for "angles" the column and the number of angles.
    """
    check, name, N, variant, log = task
    n = variant.get("n_cells", N)
    problem = _problem(name, n, variant.get("mass", SEVEN_POINT_CONE_MASS))
    scheme = variant.get("scheme", "mixed")
    options = {"angles": variant["angles"]} if "angles" in variant else {}
    solver = solver_for(scheme, n, name, **options)
    label = " ".join(f"{k}={v}" for k, v in variant.items())
    result, _ = solve(solver, problem, f"check={check} N={N} {label}", log)
    l2, linf = paper_norms(result)
    paper = PUBLISHED.get((name, "wide only" if scheme == "wide only" else "mixed"))
    k = NS.index(N)
    info = result.info
    row = {
        "check": check,
        "problem": name,
        "N": N,
        "n_cells": n,
        "scheme": scheme,
        "mass": variant.get("mass", SEVEN_POINT_CONE_MASS if name == "T21" else ""),
        "angles": info["angles"],
        "l2": f"{l2:.6e}",
        "linf": f"{linf:.6e}",
        "published_l2": published_value(paper["l2"][k]) if paper else "",
        "published_linf": published_value(paper["linf"][k]) if paper else "",
        "iterations": info["iterations"],
        "converged": info["converged"],
        "wide_fraction": info["wide_fraction"],
        "history": " ".join(f"{r:.3e}" for r in info["history"]),
        "time_s": f"{info['time']:.2f}",
    }
    print(f"check {check} {name} N={N} {label}: linf {linf:.4e}", flush=True)
    return row


CHECK_TASKS: list[tuple[str, str, int, dict[str, Any]]] = [
    *(
        ("grid", name, N, {"n_cells": n})
        for name in ("T1-square", "T2", "T20c")
        for N in (32, 64)
        for n in (N, N + 1)
    ),
    *(
        ("search", name, N, {"scheme": "everywhere"})
        for name in ("T1-square", "T2", "T20c", "T21")
        for N in (32, 64, 128)
    ),
    *(
        ("mass", "T21", N, {"mass": mass})
        for N in (32, 64, 128, 256)
        for mass in (math.pi, 4.0)
    ),
    *(
        ("angles", name, N, {"scheme": scheme, "angles": angles})
        for name, scheme in (("T1-square", "wide only"), ("T21", "mixed"))
        for N in (32, 64)
        for angles in (N // 2, N - 1, N + 1, 2 * N)
    ),
    *(
        ("start", name, N, {"scheme": "exact start"})
        for name in ("T2", "T20c")
        for N in NS
    ),
]
"""(check, problem, N, variant) of every row of chen_wan_lin2018_checks.csv."""


def cycling_row(task: tuple[str, str, int, bool]) -> dict[str, Any]:
    """One row of chen_wan_lin2018_cycling.csv: a problem, a column, n cells per side.

    The errors are those of ``evaluate``; ``first_residual`` is the residual
    at the start, the residual of the default rule at the root of
    "everywhere" in the "everywhere start" column.
    """
    name, scheme, n, log = task
    result, _ = solve(
        solver_for(scheme, n, name), PROBLEMS[name], f"N={n} {scheme}", log
    )
    info = result.info
    measured = evaluate(result)
    row = {
        "problem": name,
        "scheme": scheme,
        "n_cells": n,
        "max": f"{measured['max']:.6e}" if "max" in measured else "",
        "eigmin": f"{measured['eigmin']:.4g}",
        "u_origin": f"{origin_value(result):.6f}" if name == "T7-half" else "",
        "iterations": info["iterations"],
        "converged": info["converged"],
        "first_residual": f"{info['history'][0]:.3e}",
        "residual_max": f"{info['residual_max']:.3e}",
        "wide_fraction": info["wide_fraction"],
        "history": " ".join(f"{r:.3e}" for r in info["history"]),
        "time_s": f"{info['time']:.2f}",
    }
    print(f"cycling {name} {scheme} n={n}: converged {info['converged']}", flush=True)
    return row


CYCLING_TASKS: list[tuple[str, str, int]] = [
    ("T7-half", "everywhere start", 256),
    *(
        ("T4", scheme, n)
        for scheme in ("mixed", "everywhere")
        for n in (16, 32, 48, 64, 96, 128)
    ),
]
"""(problem, column, n) of every row of chen_wan_lin2018_cycling.csv."""


def _cost(name: str, scheme: str, N: int) -> int:
    """Rough cost of a solve, to start the longest first: a wide search costs N more."""
    wide = scheme in ("wide only", "everywhere", "everywhere start")
    wide = wide or name in ("T21", "T20c", "T7-half")
    return N * N * (N if wide else 1)


def _longest_first(costs: list[int]) -> list[int]:
    """Indices of the tasks, the most expensive first."""
    return sorted(range(len(costs)), key=lambda i: costs[i], reverse=True)


def main() -> None:
    """Parse the command line, run the solves, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=16, help="worker processes")
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--checks", action="store_true", help="write chen_wan_lin2018_checks.csv only"
    )
    which.add_argument(
        "--table5", action="store_true", help="write chen_wan_lin2018_table5.csv only"
    )
    which.add_argument(
        "--cycling", action="store_true", help="write chen_wan_lin2018_cycling.csv only"
    )
    args = parser.parse_args()
    log = not args.no_mlflow
    if args.checks:
        tasks: list[tuple[Any, ...]] = list(CHECK_TASKS)
        costs = [_cost(t[1], t[3].get("scheme", "mixed"), t[2]) for t in tasks]
        worker, path = check_row, HERE / "chen_wan_lin2018_checks.csv"
    elif args.cycling:
        tasks = list(CYCLING_TASKS)
        costs = [_cost(*t) for t in tasks]
        worker, path = cycling_row, HERE / "chen_wan_lin2018_cycling.csv"
    else:
        tasks = list(TABLE5_TASKS if args.table5 else TABLE_TASKS)
        costs = [_cost(*t) for t in tasks]
        name = "chen_wan_lin2018_table5.csv" if args.table5 else "chen_wan_lin2018.csv"
        worker, path = table_row, HERE / name
    order = _longest_first(costs)
    with Pool(args.processes) as pool:
        done = pool.map(worker, [(*tasks[i], log) for i in order], 1)
    rows = [row for _, row in sorted(zip(order, done, strict=True), key=lambda p: p[0])]
    # Written only after every solve, so no run sees a dirty checkout.
    write(path, with_rates(rows) if worker is table_row else rows)


if __name__ == "__main__":
    main()
