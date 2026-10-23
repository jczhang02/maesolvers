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

With --mean-equation the loss is the preset's divided by the number of
equation points, so its equation term is a mean (icnn.md, Finding 2); with
--boundary-weight 340000 it is the mean reading of eq. (3.8), with the
boundary weight 100 of Example 4.2.

Each seed is one MLflow run in maesolvers-icnn-<problem>, named by its seed
and the settings that differ from the preset, after the prefix --name if
given; --workers seeds train at a time, each in its own process with
THREADS_PER_SEED CPU threads (default 1), on the CPU, as for the penalty MLP
(see penalty_mlp.md, Device). Run it on a100server from the repository root::

    python benchmarks/run_icnn.py T1
    python benchmarks/run_icnn.py T6 --max-steps 5000 --out /tmp/t6-5000.csv
    python benchmarks/run_icnn.py T6 --mean-equation --boundary-weight 340000 \\
        --name diagnostic --out /tmp/t6-mean.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from run_penalty_mlp import POINTS, POINTS_SEED, band, paper_measure

from maesolvers import CATALOGUE, DirichletICNN, tracking
from maesolvers.metrics import evaluate
from maesolvers.neural import Batch, NeuralSolver, PenaltyLoss

if TYPE_CHECKING:
    import torch

HERE = Path(__file__).parent

OPTIONS = ("max_steps", "init_std", "boundary_weight", "tol", "accelerator")
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


@dataclass(frozen=True, kw_only=True)
class MeanEquationLoss(PenaltyLoss):
    """``PenaltyLoss`` divided by the number of points the equation term sums over.

    The equation term becomes a mean and the boundary term ``1 / (M + K)``
    times its weight; the logged terms other than ``loss`` stay unscaled.
    """

    def __call__(self, u: torch.nn.Module, batch: Batch) -> dict[str, torch.Tensor]:
        terms = dict(super().__call__(u, batch))
        points = batch.interior.shape[0]
        if self.equation_at_boundary:
            points += batch.boundary.shape[0]
        terms["loss"] = terms["loss"] / points
        return terms


def mean_equation(preset: DirichletICNN) -> NeuralSolver:
    """The preset with its loss replaced by :class:`MeanEquationLoss`, same weights."""
    return NeuralSolver(
        network=preset.network,
        loss=MeanEquationLoss(
            convexity_weight=0.0,
            boundary_weight=preset.boundary_weight,
            equation_at_boundary=True,
        ),
        sampler=preset.sampler,
        optimizer=preset.optimizer,
        max_steps=preset.max_steps,
        tol=preset.tol,
        seed=preset.seed,
        accelerator=preset.accelerator,
    )


def boundary_max(result, n: int = POINTS // 10) -> float:
    """Max ``|u - g|`` at ``n`` uniform random boundary points."""
    problem = result.problem
    points = problem.domain.sample_boundary(n, np.random.default_rng(POINTS_SEED))
    return float(np.max(np.abs(result.u(points) - problem.boundary.g(points))))


def run_seed(
    problem_id: str, seed: int, options: dict, name: str, mean: bool = False
) -> dict:
    """Train one seed, log it to MLflow as run ``name.format(seed)``, return its row.

    With ``mean`` the preset's loss is replaced by :class:`MeanEquationLoss`.
    """
    import mlflow
    import torch

    torch.set_num_threads(int(os.environ.get("THREADS_PER_SEED", "1")))
    problem = CATALOGUE[problem_id]
    solver = DirichletICNN(seed=seed, **options)
    if mean:
        solver = mean_equation(solver)
    logger = tracking.MLflowLogger()
    with tracking.run(f"maesolvers-icnn-{problem_id}", run_name=name.format(seed)):
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
    parser.add_argument(
        "--boundary-weight", type=float, default=DirichletICNN.boundary_weight
    )
    parser.add_argument("--tol", type=float, default=DirichletICNN.tol)
    parser.add_argument("--accelerator", default=DirichletICNN.accelerator)
    parser.add_argument(
        "--mean-equation",
        action="store_true",
        help="divide the loss by the number of equation points (icnn.md, Finding 2)",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--name", default="", help="prefix of the MLflow run names, e.g. 'diagnostic'"
    )
    args = parser.parse_args()
    options = {name: getattr(args, name) for name in OPTIONS}
    changed = {k: v for k, v in options.items() if v != getattr(DirichletICNN, k)}
    if args.mean_equation:
        changed["mean_equation"] = True
    if changed and args.out is None:
        parser.error(f"{', '.join(changed)} differ from the preset: give --out")
    label = " ".join(f"{k}={v}" for k, v in changed.items()) or "preset"
    name = " ".join(filter(None, (args.name, "seed={}", label)))
    out = args.out or HERE / f"icnn_{args.problem.lower()}.csv"
    start = time.time()
    context = get_context("spawn")  # CUDA cannot be forked
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_seed, args.problem, s, options, name, args.mean_equation)
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
    for key in FIELDS[2:8]:
        print(f"{args.problem} {key}: {band([r[key] for r in rows])}")
    stops = [r["stop"] for r in rows]
    print(f"stops: {dict((s, stops.count(s)) for s in sorted(set(stops)))}")
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
