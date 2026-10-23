"""Regenerate qs_pinn.csv: QS-PINN and its baselines on P1, T8 and T12, seeds 0 to 9.

QS-PINN has no source paper and no published table (issue #33), so this is a
comparison, not a reproduction: the QS form against the baselines its note
names (``writing/notes/qs-pinn/main.tex`` in jczhang02/monge-ampere-equations,
Section 7), on its problems P1 and P3 (Table 2):

- ``P1``: the note's P1, the unit disk with ``f = 1`` and ``g = 1/2``, whose
  convex solution is ``|x|^2 / 2`` and whose concave one ``1 - |x|^2 / 2``:
  T8 plus 1/2 (:data:`P1`, a test-local problem). The QS form starts
  there at its exact solution (``alpha = 1 / sqrt(2)``, ``b = c = 0``);
- ``T8``: the catalogue's T8, the same with ``g = 0``, solutions
  ``(|x|^2 - 1) / 2`` and ``(1 - |x|^2) / 2``, where the QS form starts
  1/2 above the solution;
- ``T12``: the note's P3, the transport of the unit disk onto the ellipse;
- ``T10``: the catalogue's ``exp(|x|^2 / 2)`` on ``(0, 1)^4``, QS-PINN only,
  with its default ``ratio = 0.6`` and with ``ratio = 0.9``: a diagnostic of
  whether the bound on ``|B|_F`` or the training sets the error there
  (``qs_pinn.md``, Finding 5), not a tuning of ``ratio``.

The methods (:data:`METHODS`) on each problem, with the same seed giving the
same collocation points to every method that shares a sampler:

- ``qs``: ``DirichletQSPINN`` on P1 and T8, ``TransportQSPINN`` on T12;
- ``mlp``: the plain MLP, the QS presets' parts with a 5 x 10 (4 x 10 on
  T12) tanh ``MLP`` as the network, behind ``IdentityStart`` on T12 as the
  ICNN is;
- ``penalty_mlp``: ``PenaltyMLP`` (not on T12: its ansatz is Dirichlet);
- ``icnn``: ``DirichletICNN`` on P1 and T8, ``TransportICNN`` on T12;
- ``icnn_quadratic``: the ICNN presets with ``ICNN(alpha=ALPHA)``, the ICNN
  plus ``ALPHA |x|^2 / 2``;
- ``qs_ratio_0.9``: ``DirichletQSPINN(ratio=0.9)``, on T10 only.

Each row holds :func:`~maesolvers.metrics.evaluate` (``max``, ``rms``,
``nonconvex_fraction``, ``eigmin``), measured on the shared evaluation grid,
and, measured by this script on the same grid:

- ``residual_max``: the largest ``|det D^2 u - f|``, and ``f_min``, the
  smallest ``f``: the two sides of the note's residual certificate
  (Corollary 5.1), on grid points, so an observation, not a proof;
- on P1 and T8, ``concave_max``, the max distance to the concave
  solution, and ``boundary_max``, the largest ``|u - g|`` at 2000 equally
  spaced boundary points;
- on T12, ``gradient_rms`` and ``gradient_max``, the error of the map
  ``grad u`` (``gradient_map_error``, by central differences), and the
  final ``into`` and ``onto`` terms of the transport loss;

then the final ``loss``, ``iterations``, ``stop``, ``time`` and ``device``,
and ``status``, ``failed`` for a seed that :func:`~maesolvers.metrics.seed_error`
counts as failed (a run that stopped as ``"diverged"`` or a non-finite error),
``ok`` otherwise.

Each seed is one MLflow run in ``maesolvers-qs_pinn-T8`` (P1 and T8,
tagged with their ``g``), ``maesolvers-qs_pinn-T12`` or
``maesolvers-qs_pinn-T10``, named by its method and seed after the prefix
--name if given, with the parameter ``method``. --workers seeds train at a
time, each in its own process with one CPU thread, on the CPU. Run it on
a100server from the repository root, writing outside the checkout so that
the runs do not see it dirty, then copy the CSV back::

    just remote-bg qs-pinn python benchmarks/run_qs_pinn.py \\
        --out /jczhang02/maesolvers-data/data/qs_pinn.csv
    python benchmarks/run_qs_pinn.py --summary   # the cells of qs_pinn.md

A run with --max-steps, --seeds, --problems or --methods differs from the
receipt and needs --out. Its rows replace the rows of the same problems and
methods in the CSV at --out, if there is one, and keep the others, so one
method can be rerun into a copy of the receipt.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from comparison import threshold, with_parts

from maesolvers import (
    ICNN,
    MLP,
    T8,
    T10,
    T12,
    Dirichlet,
    DirichletICNN,
    DirichletQSPINN,
    IdentityStart,
    PenaltyMLP,
    Problem,
    TransportICNN,
    TransportQSPINN,
    tracking,
)
from maesolvers.metrics import (
    evaluate,
    gradient_fd,
    gradient_map_error,
    seed_error,
    shared_grid,
    summarize_errors,
)
from maesolvers.neural import NeuralSolver

HERE = Path(__file__).parent
CSV = HERE / "qs_pinn.csv"
SEEDS = tuple(range(10))

ALPHA = 0.25
"""The ``alpha`` of the ICNN plus ``alpha |x|^2 / 2``: half the smallest
eigenvalue of the exact Hessians, ``I`` on T8 and ``diag(2, 1/2)`` on T12, so
the exact solutions stay representable; chosen knowing them."""

BOUNDARY_POINTS = 2000
"""Equally spaced points on the unit circle where ``boundary_max`` is taken."""


def _half(x: np.ndarray) -> np.ndarray:
    return np.full(np.shape(x)[:-1], 0.5)


def _p1_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.asarray(x) ** 2, axis=-1)


P1 = Problem(
    "T8 (g = 1/2)",
    T8.domain,
    T8.f,
    Dirichlet(_half),
    exact=_p1_exact,
    note=(
        "The QS-PINN note's P1: T8 plus 1/2, so g = 1/2 and u = |x|^2 / 2; "
        "the concave 1 - |x|^2 / 2 also solves it. Test-local, no T id of its own."
    ),
)

PROBLEMS: dict[str, Problem] = {"P1": P1, "T8": T8, "T12": T12, "T10": T10}
"""The problems by the id their rows carry."""

CONCAVE = {
    "P1": lambda x: 1.0 - _p1_exact(x),
    "T8": lambda x: 0.5 - _p1_exact(x),
}
"""The concave solutions of the Dirichlet problems."""

EXPERIMENTS = {"P1": "T8", "T8": "T8", "T12": "T12", "T10": "T10"}
"""The T id in the name of each problem's MLflow experiment."""

METHODS: dict[str, tuple[str, ...]] = {
    "P1": ("qs", "mlp", "penalty_mlp", "icnn", "icnn_quadratic"),
    "T8": ("qs", "mlp", "penalty_mlp", "icnn", "icnn_quadratic"),
    "T12": ("qs", "mlp", "icnn", "icnn_quadratic"),
    "T10": ("qs", "qs_ratio_0.9"),
}
"""The methods on each problem, in the order of the tables."""

FIELDS = (
    "problem",
    "method",
    "seed",
    "status",
    "max",
    "rms",
    "nonconvex_fraction",
    "eigmin",
    "residual_max",
    "f_min",
    "concave_max",
    "boundary_max",
    "gradient_rms",
    "gradient_max",
    "into",
    "onto",
    "loss",
    "iterations",
    "stop",
    "time",
    "device",
)


def solver(problem_id: str, method: str, seed: int, max_steps: int | None) -> Any:
    """The solver of a method on a problem, for one seed."""
    steps = {} if max_steps is None else {"max_steps": max_steps}
    if method == "qs_ratio_0.9":
        return DirichletQSPINN(ratio=0.9, seed=seed, **steps)
    if problem_id != "T12":
        if method == "penalty_mlp":
            return PenaltyMLP(seed=seed, **steps)
        if method == "qs":
            return DirichletQSPINN(seed=seed, **steps)
        icnn = DirichletICNN(seed=seed, **steps)
        networks = {
            "icnn": None,
            "mlp": MLP(width=10, depth=5, activation="tanh"),
            "icnn_quadratic": ICNN(alpha=ALPHA),
        }
    else:
        if method == "qs":
            return TransportQSPINN(seed=seed, **steps)
        icnn = TransportICNN(seed=seed, **steps)
        networks = {
            "icnn": None,
            "mlp": IdentityStart(network=MLP(width=10, depth=4, activation="tanh")),
            "icnn_quadratic": IdentityStart(network=ICNN(depth=4, alpha=ALPHA)),
        }
    network = networks[method]
    return icnn if network is None else with_parts(icnn, network=network)


def circle(n: int) -> np.ndarray:
    """``n`` equally spaced points on the unit circle."""
    t = 2.0 * np.pi * np.arange(n) / n
    return np.stack([np.cos(t), np.sin(t)], axis=-1)


def measures(problem_id: str, result) -> dict[str, float]:
    """The quantities beyond :func:`evaluate` that a row holds (see the module)."""
    problem = PROBLEMS[problem_id]
    grid = shared_grid(problem.domain)
    hess = np.asarray(result.hessian(grid))
    f = problem.f(grid)
    out = {
        "residual_max": float(np.max(np.abs(np.linalg.det(hess) - f))),
        "f_min": float(np.min(f)),
    }
    if problem_id in CONCAVE:
        assert isinstance(problem.boundary, Dirichlet)
        u = result.u(grid)
        edge = circle(BOUNDARY_POINTS)
        out["concave_max"] = float(np.max(np.abs(u - CONCAVE[problem_id](grid))))
        misfit = result.u(edge) - problem.boundary.g(edge)
        out["boundary_max"] = float(np.max(np.abs(misfit)))
    elif problem_id == "T12":
        error = gradient_map_error(
            gradient_fd(result.u, grid), gradient_fd(T12.exact, grid)
        )
        out |= error
        out |= {key: result.info["loss_terms"][key] for key in ("into", "onto")}
    return out


def run_seed(task: tuple[str, str, int, int | None, str]) -> dict[str, Any]:
    """Train one method on one problem for one seed, log it, return its row."""
    import mlflow
    import torch

    problem_id, method, seed, max_steps, name = task
    torch.set_num_threads(1)
    solving: NeuralSolver = solver(problem_id, method, seed, max_steps)
    problem = PROBLEMS[problem_id]
    logger = tracking.MLflowLogger()
    run_name = " ".join(filter(None, (name, f"{method} seed={seed}")))
    tags = {"P1": {"g": "0.5"}, "T8": {"g": "0"}}.get(problem_id, {})
    experiment = f"maesolvers-qs_pinn-{EXPERIMENTS[problem_id]}"
    with tracking.run(experiment, run_name=run_name, tags=tags):
        result = solving.solve(problem, callbacks=[logger])
        extra = measures(problem_id, result)
        mlflow.log_params({"method": method})
        mlflow.log_metrics({k: v for k, v in extra.items() if np.isfinite(v)})
    metrics = logger.evaluation or evaluate(result)
    info = result.info
    return {
        "problem": problem_id,
        "method": method,
        "seed": seed,
        "status": "failed" if seed_error(result) is None else "ok",
        **{key: metrics[key] for key in ("max", "rms", "nonconvex_fraction", "eigmin")},
        **extra,
        "loss": info["loss_terms"]["loss"],
        "iterations": info["iterations"],
        "stop": info["stop"],
        "time": info["time"],
        "device": info["device"],
    }


def read(path: Path) -> list[dict[str, str]]:
    """The rows of a receipt, as strings."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def band(values: list[float]) -> str:
    """Median [5%, 95%] of finite values, three significant digits."""
    s = summarize_errors(values, threshold=np.inf)
    return f"{s.median:.2e} [{s.q05:.2e}, {s.q95:.2e}]"


def merge(old: list[dict[str, str]], new: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows of ``new``, and those of ``old`` whose problem and method ``new`` lacks.

    In the order of :data:`PROBLEMS` and :data:`METHODS`, then by seed.
    """
    ran = {(r["problem"], r["method"]) for r in new}
    rows = [r for r in old if (r["problem"], r["method"]) not in ran] + new
    problems = list(PROBLEMS)

    def key(row: dict[str, str]) -> tuple[int, int, int]:
        problem = row["problem"]
        method = METHODS[problem].index(row["method"])
        return problems.index(problem), method, int(row["seed"])

    return sorted(rows, key=key)


def summary(path: Path) -> None:
    """Print the cells of qs_pinn.md from a receipt, one line per problem and method.

    The max error as the comparison table gives it (median [5%, 95%], the
    successes below 1% of the exact oscillation, the failed seeds by
    number), then the medians [5%, 95%] of the other measures over the
    seeds that did not fail, the smallest ``eigmin`` over them, the seeds
    that are not convex on the grid, that pass the residual certificate on
    the grid, and on P1 and T8 that lie on the concave solution (its max
    distance below the threshold), then the stops with the seeds and the
    range of the iterations of each.
    """
    rows = read(path)
    for problem_id, methods in METHODS.items():
        problem = PROBLEMS[problem_id]
        for method in methods:
            mine = [
                r for r in rows if (r["problem"], r["method"]) == (problem_id, method)
            ]
            if not mine:
                continue
            errors = [
                None if r["status"] == "failed" else float(r["max"]) for r in mine
            ]
            s = summarize_errors(errors, threshold(problem))
            ok = [r for r in mine if r["status"] != "failed"]
            failed = [r["seed"] for r in mine if r["status"] == "failed"]
            nonconvex = [r["seed"] for r in ok if float(r["nonconvex_fraction"]) > 0]
            certified = [
                r["seed"] for r in ok if float(r["residual_max"]) < float(r["f_min"])
            ]
            line = (
                f"{problem_id} {method}: max {s.median:.2e} [{s.q05:.2e}, {s.q95:.2e}], "
                f"{round(s.success_rate * s.seeds)}/{s.seeds} below the threshold, "
                f"failed {failed or 'none'}; "
            )
            keys = [
                "rms",
                "nonconvex_fraction",
                "eigmin",
                "residual_max",
                "loss",
                "time",
            ]
            if problem_id in CONCAVE:
                keys += ["concave_max", "boundary_max"]
            elif problem_id == "T12":
                keys += ["gradient_rms", "gradient_max", "into", "onto"]
            line += "; ".join(
                f"{k} {band([float(r[k]) for r in ok])}" for k in keys if ok
            )
            if ok:
                line += f"; smallest eigmin {min(float(r['eigmin']) for r in ok):.2e}"
            line += f"; non-convex seeds {nonconvex or 'none'}"
            line += f"; certified seeds {certified or 'none'}"
            if problem_id in CONCAVE:
                limit = threshold(problem)
                concave = [r["seed"] for r in ok if float(r["concave_max"]) < limit]
                line += f"; on the concave solution {concave or 'none'}"
            stops = []
            for stop in sorted({r["stop"] for r in mine}):
                these = [r for r in mine if r["stop"] == stop]
                steps = [int(r["iterations"]) for r in these]
                seeds = ", ".join(r["seed"] for r in these)
                stops.append(
                    f"{stop} {len(these)} (seeds {seeds}; "
                    f"steps {min(steps)} to {max(steps)})"
                )
            print(line + "; stops " + ", ".join(stops))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--name", default="", help="prefix of the MLflow run names, e.g. 'diagnostic'"
    )
    parser.add_argument(
        "--summary", action="store_true", help="print the cells of qs_pinn.md"
    )
    args = parser.parse_args()
    if args.summary:
        summary(args.out or CSV)
        return
    changed = (
        args.problems is not None
        or args.methods is not None
        or args.seeds != list(SEEDS)
        or args.max_steps is not None
    )
    if changed and args.out is None:
        parser.error("--problems, --methods, --seeds or --max-steps differ: give --out")
    tasks = [
        (problem_id, method, seed, args.max_steps, args.name)
        for problem_id in args.problems or PROBLEMS
        for method in METHODS[problem_id]
        if args.methods is None or method in args.methods
        for seed in args.seeds
    ]
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"  # inherited by the spawned workers
    start = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(run_seed, tasks))
    out = args.out or CSV
    new = [
        {k: f"{v:.6e}" if isinstance(v, float) else str(v) for k, v in row.items()}
        for row in rows
    ]
    rows = merge(read(out) if out.exists() else [], new)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary(out)
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
