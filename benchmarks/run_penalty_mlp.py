"""Regenerate penalty_mlp_<problem>.csv: PenaltyMLP over seeds 0 to 9.

Trains the penalty MLP of Nystrom-Vestberg 2023 with the preset's defaults
(``PenaltyMLP()``) on a catalogue problem, one seed per process, and writes
one row per seed next to this file; penalty_mlp.md reads the CSVs. Each row
holds these measures of the same network:

- ``grid_max``, ``grid_rms``: :func:`~maesolvers.metrics.evaluate`, the
  shared evaluation grid of ADR 0007, the primary error of the library;
- ``points_max``, ``points_mean``: the paper's measure, max and mean
  absolute error at 10^6 uniform random points inside the domain, drawn
  once from the fixed generator seed POINTS_SEED (Section 4.6 of the paper);
- ``boundary_max``: the max absolute error at 10^5 uniform random points on
  the boundary, where ``u - g = (G - g) + D N``; the next three columns
  split it at the same points: ``lift_misfit_max`` (max ``|G - g|``),
  ``distance_boundary_max`` (max ``|D|``) and ``leak_max`` (max ``|D N|``);
- ``lift_iterations``, ``distance_iterations``, ``fits_converged``: how the
  fits of ``G`` and ``D`` ended (``Result.info["network"]``).

Each seed is one MLflow run in maesolvers-penalty-mlp-<problem>, named by
its seed and the settings that differ from the preset; --workers seeds
train at a time, each in its own process with THREADS_PER_SEED CPU threads
(default 1). The receipts reproduce only with one thread per seed: the
rounding of the fits, and so the result, depends on the thread count. They
train on the CPU: the network is so small that a training step is bound by
kernel launches, and on a100server one CPU thread (about 70 ms per step)
beats an A100 shared by four or more processes (about 200 ms; see
penalty_mlp.md). Run it on a100server from the repository root::

    python benchmarks/run_penalty_mlp.py T6
    python benchmarks/run_penalty_mlp.py T1 --fit-tol 1e-7 --out /tmp/t1-fit.csv
    python benchmarks/run_penalty_mlp.py T1 --optimizer bfgs --out /tmp/t1-bfgs.csv
    CUDA_VISIBLE_DEVICES=0 python benchmarks/run_penalty_mlp.py T6 --accelerator gpu --out t6.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from maesolvers import CATALOGUE, PenaltyMLP, tracking
from maesolvers.metrics import evaluate, seed_error, summarize_errors
from maesolvers.neural import BFGS, NeuralSolver, sample_interior

HERE = Path(__file__).parent
POINTS = 10**6
POINTS_SEED = 2023
"""Seed of the 10^6 evaluation points, the same for every seed and problem."""

OPTIONS = ("max_steps", "fit_steps", "fit_tol", "accelerator")
"""Preset settings the command line can change; a change needs --out."""

OPTIMIZERS = ("lbfgs", "bfgs")
"""--optimizer: the preset's L-BFGS, or SciPy's dense BFGS of DirichletICNN."""

FIELDS = (
    "problem",
    "seed",
    "grid_max",
    "grid_rms",
    "points_max",
    "points_mean",
    "boundary_max",
    "lift_misfit_max",
    "distance_boundary_max",
    "leak_max",
    "nonconvex_fraction",
    "iterations",
    "stop",
    "loss",
    "lift_iterations",
    "distance_iterations",
    "fits_converged",
    "time",
    "device",
    "threads",
)


def paper_measure(result, n: int = POINTS) -> tuple[float, float]:
    """Max and mean absolute error at ``n`` uniform random interior points."""
    problem = result.problem
    points = sample_interior(problem.domain, n, np.random.default_rng(POINTS_SEED))
    error = np.abs(result.u(points) - problem.exact(points))
    return float(np.max(error)), float(np.mean(error))


def boundary_split(solver, result, n: int = POINTS // 10) -> dict[str, float]:
    """Max ``|u - g|``, ``|G - g|``, ``|D|`` and ``|D N|`` at ``n`` boundary points.

    ``G`` and ``D`` are rebuilt by ``solver.setup``, which repeats the fits
    from the seed in this process and so gives the networks of the solve;
    the check against the solve's fit report guards this.
    """
    import torch

    problem = result.problem
    points = problem.domain.sample_boundary(n, np.random.default_rng(POINTS_SEED))
    model = solver.setup(problem).model
    assert model.report["lift"] == result.info["network"]["lift"], "G differs"
    assert model.report["distance"] == result.info["network"]["distance"]
    with torch.no_grad():
        x = torch.tensor(points, dtype=torch.float64)
        lift = model.lift(x).numpy()
        distance = model.distance(x).numpy()
    u, g = result.u(points), problem.boundary.g(points)
    return {
        "boundary_max": float(np.max(np.abs(u - g))),
        "lift_misfit_max": float(np.max(np.abs(lift - g))),
        "distance_boundary_max": float(np.max(np.abs(distance))),
        "leak_max": float(np.max(np.abs(u - lift))),
    }


def with_bfgs(preset: PenaltyMLP) -> NeuralSolver:
    """The preset's parts and settings with the optimizer ``BFGS()``."""
    return NeuralSolver(
        network=preset.network,
        loss=preset.loss,
        sampler=preset.sampler,
        optimizer=BFGS(),
        max_steps=preset.max_steps,
        tol=preset.tol,
        seed=preset.seed,
        accelerator=preset.accelerator,
    )


def run_seed(
    problem_id: str, seed: int, options: dict, label: str, optimizer: str = "lbfgs"
) -> dict:
    """Train one seed, log it to MLflow, and return its CSV row.

    With ``optimizer="bfgs"`` the preset trains with :class:`BFGS` (the fits
    of ``G`` and ``D`` keep their L-BFGS). The row also holds ``failed``, not
    a CSV column: whether the seed failed by the rule of
    :func:`~maesolvers.metrics.seed_error`.
    """
    import mlflow
    import torch

    torch.set_num_threads(int(os.environ.get("THREADS_PER_SEED", "1")))
    problem = CATALOGUE[problem_id]
    solver = PenaltyMLP(seed=seed, **options)
    if optimizer == "bfgs":
        solver = with_bfgs(solver)
    logger = tracking.MLflowLogger()
    with tracking.run(
        f"maesolvers-penalty-mlp-{problem_id}", run_name=f"seed={seed} {label}"
    ):
        result = solver.solve(problem, callbacks=[logger])
        info, fits = result.info, result.info["network"]
        points_max, points_mean = paper_measure(result)
        split = boundary_split(solver, result)
        mlflow.log_params({"device": info["device"], "threads": info["threads"]})
        mlflow.log_metrics(
            {
                "points_max": points_max,
                "points_mean": points_mean,
                **split,
                "lift_iterations": fits["lift"]["iterations"],
                "distance_iterations": fits["distance"]["iterations"],
                "lift_max_gradient": fits["lift"]["max_gradient"],
                "distance_max_gradient": fits["distance"]["max_gradient"],
            }
        )
    metrics = logger.evaluation or evaluate(result)
    return {
        "problem": problem_id,
        "seed": seed,
        "grid_max": metrics["max"],
        "grid_rms": metrics["rms"],
        "points_max": points_max,
        "points_mean": points_mean,
        **split,
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        "iterations": info["iterations"],
        "stop": info["stop"],
        "loss": info["loss_terms"]["loss"],
        "lift_iterations": fits["lift"]["iterations"],
        "distance_iterations": fits["distance"]["iterations"],
        "fits_converged": fits["lift"]["converged"] and fits["distance"]["converged"],
        "time": info["time"],
        "device": info["device"],
        "threads": info["threads"],
        "failed": seed_error(result) is None,
    }


def band(values: list[float | None]) -> str:
    """Median and 5% to 95% quantiles over the seeds, a failed seed as inf (ADR 0007)."""
    s = summarize_errors(values, threshold=np.inf)
    return f"{s.median:.3e} [{s.q05:.3e}, {s.q95:.3e}], {s.failed} failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", choices=sorted(CATALOGUE))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-steps", type=int, default=PenaltyMLP.max_steps)
    parser.add_argument("--fit-steps", type=int, default=PenaltyMLP.fit_steps)
    parser.add_argument("--fit-tol", type=float, default=PenaltyMLP.fit_tol)
    parser.add_argument("--accelerator", default=PenaltyMLP.accelerator)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="lbfgs")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    options = {name: getattr(args, name) for name in OPTIONS}
    changed = {k: v for k, v in options.items() if v != getattr(PenaltyMLP, k)}
    if args.optimizer != "lbfgs":
        changed["optimizer"] = args.optimizer
    if changed and args.out is None:
        parser.error(f"{', '.join(changed)} differ from the preset: give --out")
    label = " ".join(f"{k}={v}" for k, v in changed.items()) or "preset"
    out = args.out or HERE / f"penalty_mlp_{args.problem.lower()}.csv"
    start = time.time()
    context = get_context("spawn")  # CUDA cannot be forked
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_seed, args.problem, s, options, label, args.optimizer)
            for s in args.seeds
        ]
        rows = sorted((f.result() for f in futures), key=lambda r: r["seed"])
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: f"{v:.6e}" if isinstance(v, float) else v
                    for k, v in row.items()
                    if k in FIELDS
                }
            )
    for key in FIELDS[2:10]:
        values = [None if r["failed"] else r[key] for r in rows]
        print(f"{args.problem} {key}: {band(values)}")
    stops = [r["stop"] for r in rows]
    print(f"stops: {dict((s, stops.count(s)) for s in sorted(set(stops)))}")
    print(f"fits converged: {sum(r['fits_converged'] for r in rows)} of {len(rows)}")
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
