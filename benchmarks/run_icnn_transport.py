"""Regenerate icnn_t12.csv: TransportICNN on T12 over seeds 0 to 9, Adam and L-BFGS.

The reproduction of Caboussat-Peruso 2025, Table 2 (arXiv 2501.10162v2):
the input convex network on T12, the transport of the unit disk onto an
ellipse, trained with two schedules, each over ten seeds:

- ``lbfgs``: the preset's defaults, Adam for 200 steps then L-BFGS
  (Table 2, "L-BFGS"; Figure 5, "Adam - L-BFGS");
- ``adam``: the same with Adam for every step (Table 2, "Adam"; Figure 5,
  "Adam only"), ``adam_steps = max_steps``.

The same seed gives both schedules the same collocation points and the same
initial network. Each row holds these measures of one trained network:

- ``train_rms``, ``train_max``: the gradient-map error (``gradient_map_error``)
  at the seed's 800 interior collocation points, the paper's "train" points;
- ``test_rms``, ``test_max``: the same on the shared evaluation grid of ADR
  0007 (the 201 x 201 grid over the bounding box, points in the disk), our
  "test" points;
- ``grid_max``, ``nonconvex_fraction``: :func:`~maesolvers.metrics.evaluate`,
  the error of ``u`` up to a constant and the share of non-convex points;
- ``loss`` and its terms at the end, ``loss_switch``, the loss at the
  start of the step after the Adam phase of the ``lbfgs`` schedule (the
  dashed line of Figure 5), and how the training and the identity fit ended.

``gradient_rms`` is the root mean square over the points; the L2 norm over
the disk is about ``sqrt(pi)`` times it (icnn.md, T12, Conventions). The
gradients are central differences of ``result.u`` and of the exact
solution (``gradient_fd``).

Each seed is one MLflow run in maesolvers-icnn-T12, named by its schedule,
its seed and the settings that differ from the preset, after the prefix
--name if given; --workers seeds train at a time, each in its own process
with one CPU thread, on the device of --accelerator (a100server: a GPU
picked with CUDA_VISIBLE_DEVICES). Run it on a100server from the
repository root::

    CUDA_VISIBLE_DEVICES=2 python benchmarks/run_icnn_transport.py
    python benchmarks/run_icnn_transport.py --schedules lbfgs --seeds 0 1 \\
        --adam-lr 1e-2 --name diagnostic --out /tmp/t12.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from run_penalty_mlp import band

from maesolvers import T12, TransportICNN, tracking
from maesolvers.metrics import evaluate, gradient_fd, gradient_map_error, shared_grid

HERE = Path(__file__).parent
CSV = HERE / "icnn_t12.csv"

SCHEDULES = ("lbfgs", "adam")
"""Table 2's rows: Adam then L-BFGS (the preset), and Adam alone."""

OPTIONS = ("init_std", "fit_steps", "adam_lr", "adam_steps", "max_steps", "accelerator")
"""Preset settings the command line can change; a change needs --out."""

FIELDS = (
    "schedule",
    "seed",
    "train_rms",
    "train_max",
    "test_rms",
    "test_max",
    "grid_max",
    "nonconvex_fraction",
    "loss",
    "equation",
    "into",
    "onto",
    "loss_switch",
    "iterations",
    "stop",
    "fit_iterations",
    "fit_max_misfit",
    "time",
    "device",
)


def gradient_errors(result, points: np.ndarray) -> dict[str, float]:
    """``gradient_map_error`` of a result on T12 at the points, by central differences."""
    return gradient_map_error(
        gradient_fd(result.u, points), gradient_fd(T12.exact, points)
    )


def solver(schedule: str, seed: int, options: dict) -> TransportICNN:
    """The preset with the options, Adam for every step under the ``adam`` schedule."""
    preset = TransportICNN(seed=seed, **options)
    if schedule == "adam":
        return TransportICNN(seed=seed, **{**options, "adam_steps": preset.max_steps})
    return preset


def run_seed(schedule: str, seed: int, options: dict, name: str) -> dict:
    """Train one seed under one schedule, log it to MLflow, return its row."""
    import mlflow
    import torch

    torch.set_num_threads(1)
    preset = solver(schedule, seed, options)
    logger = tracking.MLflowLogger()
    with tracking.run("maesolvers-icnn-T12", run_name=name.format(schedule, seed)):
        result = preset.solve(T12, callbacks=[logger])
        train = gradient_errors(result, preset.collocation(T12).interior)
        test = gradient_errors(result, shared_grid(T12.domain))
        mlflow.log_params({"schedule": schedule, "device": result.info["device"]})
        mlflow.log_metrics(
            {
                "train_rms": train["gradient_rms"],
                "train_max": train["gradient_max"],
                "test_rms": test["gradient_rms"],
                "test_max": test["gradient_max"],
            }
        )
    metrics = logger.evaluation or evaluate(result)
    info = result.info
    history, switch = info["history"], preset.adam_steps
    fit = info["network"]["identity_fit"]
    return {
        "schedule": schedule,
        "seed": seed,
        "train_rms": train["gradient_rms"],
        "train_max": train["gradient_max"],
        "test_rms": test["gradient_rms"],
        "test_max": test["gradient_max"],
        "grid_max": metrics["max"],
        "nonconvex_fraction": metrics["nonconvex_fraction"],
        **{
            key: info["loss_terms"][key] for key in ("loss", "equation", "into", "onto")
        },
        "loss_switch": history[switch] if switch < len(history) else "",
        "iterations": info["iterations"],
        "stop": info["stop"],
        "fit_iterations": fit["iterations"],
        "fit_max_misfit": fit["max_misfit"],
        "time": info["time"],
        "device": info["device"],
    }


def mean_std(values: list[float]) -> str:
    """Mean and sample standard deviation, as Table 2 prints them."""
    return f"{np.mean(values):.3e} ({np.std(values, ddof=1):.3e})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedules", nargs="+", choices=SCHEDULES, default=SCHEDULES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--init-std", type=float, default=TransportICNN.init_std)
    parser.add_argument("--fit-steps", type=int, default=TransportICNN.fit_steps)
    parser.add_argument("--adam-lr", type=float, default=TransportICNN.adam_lr)
    parser.add_argument("--adam-steps", type=int, default=TransportICNN.adam_steps)
    parser.add_argument("--max-steps", type=int, default=TransportICNN.max_steps)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--name", default="", help="prefix of the MLflow run names, e.g. 'diagnostic'"
    )
    args = parser.parse_args()
    options = {name: getattr(args, name) for name in OPTIONS}
    changed = {
        k: v
        for k, v in options.items()
        if k != "accelerator" and v != getattr(TransportICNN, k)
    }
    if (changed or args.seeds != list(range(10))) and args.out is None:
        parser.error(
            f"{', '.join(changed) or 'seeds'} differ from the receipt: give --out"
        )
    label = " ".join(f"{k}={v}" for k, v in changed.items()) or "preset"
    name = " ".join(filter(None, (args.name, "{} seed={}", label)))
    out = args.out or CSV
    start = time.time()
    context = get_context("spawn")  # CUDA cannot be forked
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_seed, schedule, seed, options, name)
            for schedule in args.schedules
            for seed in args.seeds
        ]
        rows = [f.result() for f in futures]
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.6e}" if isinstance(v, float) else v for k, v in row.items()}
            )
    for schedule in args.schedules:
        mine = [r for r in rows if r["schedule"] == schedule]
        for key in ("loss", "train_rms", "test_rms", "test_max", "grid_max"):
            values = [r[key] for r in mine]
            print(f"{schedule} {key}: {band(values)}; mean (std) {mean_std(values)}")
        stops = [r["stop"] for r in mine]
        print(
            f"{schedule} stops: {dict((s, stops.count(s)) for s in sorted(set(stops)))}"
        )
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
