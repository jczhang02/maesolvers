"""Regenerate icnn_<problem>.csv: DirichletICNN over seeds 0 to 9.

Trains the input convex network of Nystrom-Vestberg 2023 with the preset's
defaults (``DirichletICNN()``) on a catalogue problem, one seed per process,
and writes one row per seed next to this file; icnn.md reads the CSVs. Each
row holds these measures of the same network:

- ``grid_max``, ``grid_rms``: :func:`~maesolvers.metrics.evaluate`, the
  shared evaluation grid of ADR 0007, the primary error of the library;
- ``points_max``, ``points_mean``: the paper's measure, max and mean
  absolute error at 10^6 uniform random points inside the domain, the same
  points as in run_penalty_mlp.py (Section 4.6 of the paper);
- ``boundary_max``: the max absolute error at 10^5 uniform random points on
  the boundary, where the loss imposes the boundary condition by a penalty;
- ``nonconvex_fraction``, ``eigmin``: from ``evaluate``, on the shared grid;
- ``iterations``, ``stop`` and the loss terms of the trained network.

Each seed is one MLflow run in maesolvers-icnn-<problem>, named by its seed
and the settings that differ from the preset; --workers seeds train at a
time, each in its own process with THREADS_PER_SEED CPU threads (default
1), on the CPU, as for the penalty MLP (see penalty_mlp.md, Device). Run it
on a100server from the repository root::

    python benchmarks/run_icnn.py T1
    python benchmarks/run_icnn.py T6 --max-steps 5000 --out /tmp/t6-5000.csv
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
from run_penalty_mlp import POINTS, POINTS_SEED, band, paper_measure

from maesolvers import CATALOGUE, DirichletICNN, tracking
from maesolvers.metrics import evaluate

HERE = Path(__file__).parent

OPTIONS = ("max_steps", "init_std", "accelerator")
"""Preset settings the command line can change; a change needs --out."""

FIELDS = (
    "problem",
    "seed",
    "grid_max",
    "grid_rms",
    "points_max",
    "points_mean",
    "boundary_max",
    "nonconvex_fraction",
    "eigmin",
    "iterations",
    "stop",
    "loss",
    "equation",
    "boundary",
    "time",
    "device",
    "threads",
)


def boundary_max(result, n: int = POINTS // 10) -> float:
    """Max ``|u - g|`` at ``n`` uniform random boundary points."""
    problem = result.problem
    points = problem.domain.sample_boundary(n, np.random.default_rng(POINTS_SEED))
    return float(np.max(np.abs(result.u(points) - problem.boundary.g(points))))


def run_seed(problem_id: str, seed: int, options: dict, label: str) -> dict:
    """Train one seed, log it to MLflow, and return its CSV row."""
    import mlflow
    import torch

    torch.set_num_threads(int(os.environ.get("THREADS_PER_SEED", "1")))
    problem = CATALOGUE[problem_id]
    solver = DirichletICNN(seed=seed, **options)
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-icnn-{problem_id}", run_name=f"seed={seed} {label}"):
        result = solver.solve(problem, callbacks=[logger])
        info = result.info
        points_max, points_mean = paper_measure(result)
        edge = boundary_max(result)
        mlflow.log_params({"device": info["device"], "threads": info["threads"]})
        mlflow.log_metrics(
            {"points_max": points_max, "points_mean": points_mean, "boundary_max": edge}
        )
    metrics = logger.evaluation or evaluate(result)
    terms = info["loss_terms"]
    return {
        "problem": problem_id,
        "seed": seed,
        "grid_max": metrics["max"],
        "grid_rms": metrics["rms"],
        "points_max": points_max,
        "points_mean": points_mean,
        "boundary_max": edge,
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        "eigmin": metrics["eigmin"],
        "iterations": info["iterations"],
        "stop": info["stop"],
        "loss": terms["loss"],
        "equation": terms["equation"],
        "boundary": terms["boundary"],
        "time": info["time"],
        "device": info["device"],
        "threads": info["threads"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", choices=sorted(CATALOGUE))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-steps", type=int, default=DirichletICNN.max_steps)
    parser.add_argument("--init-std", type=float, default=DirichletICNN.init_std)
    parser.add_argument("--accelerator", default=DirichletICNN.accelerator)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    options = {name: getattr(args, name) for name in OPTIONS}
    changed = {k: v for k, v in options.items() if v != getattr(DirichletICNN, k)}
    if changed and args.out is None:
        parser.error(f"{', '.join(changed)} differ from the preset: give --out")
    label = " ".join(f"{k}={v}" for k, v in changed.items()) or "preset"
    out = args.out or HERE / f"icnn_{args.problem.lower()}.csv"
    start = time.time()
    context = get_context("spawn")  # CUDA cannot be forked
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_seed, args.problem, s, options, label) for s in args.seeds
        ]
        rows = sorted((f.result() for f in futures), key=lambda r: r["seed"])
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.6e}" if isinstance(v, float) else v for k, v in row.items()}
            )
    for key in FIELDS[2:8]:
        print(f"{args.problem} {key}: {band([r[key] for r in rows])}")
    stops = [r["stop"] for r in rows]
    print(f"stops: {dict((s, stops.count(s)) for s in sorted(set(stops)))}")
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
