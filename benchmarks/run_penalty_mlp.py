"""Regenerate penalty_mlp_<problem>.csv: PenaltyMLP over seeds 0 to 9.

Trains the penalty MLP of Nystrom-Vestberg 2023 with the paper's defaults
(``PenaltyMLP()``) on a catalogue problem, one seed per process, and writes
one row per seed next to this file; penalty_mlp.md reads the CSVs. Each row
holds two measures of the same network:

- ``grid_max``, ``grid_rms``: :func:`~maesolvers.metrics.evaluate`, the
  shared evaluation grid of ADR 0007, the primary error of the library;
- ``points_max``, ``points_mean``: the paper's measure, max and mean
  absolute error at 10^6 uniform random points inside the domain, drawn
  once from the fixed generator seed POINTS_SEED (Section 4.6 of the paper);
- ``boundary_max``: the max absolute error at 10^5 uniform random points on
  the boundary, where the ansatz ``G + D N`` reduces to the fit ``G`` of the
  boundary data.

Each seed is one MLflow run in maesolvers-penalty-mlp-<problem>, named by
its seed and device; --workers seeds train at a time, each in its own
process with one CPU thread. The receipts train on the CPU: the network is
so small that a training step is bound by kernel launches, and on
a100server one CPU thread (about 70 ms per step) beats an A100 shared by
four or more processes (about 200 ms; see penalty_mlp.md). Run it on
a100server from the repository root::

    python benchmarks/run_penalty_mlp.py T6
    CUDA_VISIBLE_DEVICES=0 python benchmarks/run_penalty_mlp.py T6 --accelerator gpu
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
from maesolvers.metrics import evaluate
from maesolvers.neural import sample_interior

HERE = Path(__file__).parent
POINTS = 10**6
POINTS_SEED = 2023
"""Seed of the 10^6 evaluation points, the same for every seed and problem."""

FIELDS = (
    "problem",
    "seed",
    "grid_max",
    "grid_rms",
    "points_max",
    "points_mean",
    "boundary_max",
    "nonconvex_fraction",
    "iterations",
    "stop",
    "loss",
    "time",
    "device",
)


def paper_measure(result, n: int = POINTS) -> tuple[float, float]:
    """Max and mean absolute error at ``n`` uniform random interior points."""
    problem = result.problem
    points = sample_interior(problem.domain, n, np.random.default_rng(POINTS_SEED))
    error = np.abs(result.u(points) - problem.exact(points))
    return float(np.max(error)), float(np.mean(error))


def boundary_error(result, n: int = POINTS // 10) -> float:
    """Max absolute error at ``n`` uniform random boundary points."""
    problem = result.problem
    rng = np.random.default_rng(POINTS_SEED)
    points = problem.domain.sample_boundary(n, rng)
    return float(np.max(np.abs(result.u(points) - problem.exact(points))))


def run_seed(problem_id: str, seed: int, max_steps: int, accelerator: str) -> dict:
    """Train one seed, log it to MLflow, and return its CSV row."""
    import torch

    torch.set_num_threads(int(os.environ.get("THREADS_PER_SEED", "1")))
    problem = CATALOGUE[problem_id]
    solver = PenaltyMLP(seed=seed, max_steps=max_steps, accelerator=accelerator)
    logger = tracking.MLflowLogger()
    with tracking.run(
        f"maesolvers-penalty-mlp-{problem_id}", run_name=f"seed={seed} {accelerator}"
    ):
        result = solver.solve(problem, callbacks=[logger])
        points_max, points_mean = paper_measure(result)
        boundary_max = boundary_error(result)
        import mlflow

        mlflow.log_metrics(
            {
                "points_max": points_max,
                "points_mean": points_mean,
                "boundary_max": boundary_max,
            }
        )
    metrics = logger.evaluation or evaluate(result)
    info = result.info
    return {
        "problem": problem_id,
        "seed": seed,
        "grid_max": metrics["max"],
        "grid_rms": metrics["rms"],
        "points_max": points_max,
        "points_mean": points_mean,
        "boundary_max": boundary_max,
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        "iterations": info["iterations"],
        "stop": info["stop"],
        "loss": info["loss_terms"]["loss"],
        "time": info["time"],
        "device": info["device"],
    }


def band(values: list[float]) -> str:
    """Median and 5% to 95% quantiles, as ADR 0007 reports seeds."""
    q05, q50, q95 = np.quantile(np.asarray(values, dtype=float), [0.05, 0.5, 0.95])
    return f"{q50:.3e} [{q05:.3e}, {q95:.3e}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", choices=sorted(CATALOGUE))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-steps", type=int, default=PenaltyMLP.max_steps)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or HERE / f"penalty_mlp_{args.problem.lower()}.csv"
    start = time.time()
    context = get_context("spawn")  # CUDA cannot be forked
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_seed, args.problem, s, args.max_steps, args.accelerator)
            for s in args.seeds
        ]
        rows = sorted((f.result() for f in futures), key=lambda r: r["seed"])
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.6e}" if isinstance(v, float) else v for k, v in row.items()}
            )
    for key in ("grid_max", "grid_rms", "points_max", "points_mean", "boundary_max"):
        print(f"{args.problem} {key}: {band([r[key] for r in rows])}")
    stops = [r["stop"] for r in rows]
    print(f"stops: {dict((s, stops.count(s)) for s in sorted(set(stops)))}")
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
