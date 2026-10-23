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

Two modes train nothing and log nothing. ``--summary`` prints the cells of
the T12 tables in icnn.md from the receipt (or --out): per schedule and
column the mean (std) as Table 2 prints them, the median [5%, 95%] over the
seeds, and a bootstrap 5% to 95% band of the mean over the seeds, all to
three significant digits. ``--floor`` prints, for each seed of --seeds, the
transport term of eq. (3.4) that T12's exact map leaves at the preset's
collocation points (icnn.md, Finding 3).
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from comparison import with_parts

from maesolvers import T12, Problem, TransportICNN, TransportSampler, tracking
from maesolvers.core import Disk, Ellipse, Transport
from maesolvers.metrics import (
    evaluate,
    gradient_fd,
    gradient_map_error,
    shared_grid,
    summarize_errors,
)
from maesolvers.neural import Collocation, NeuralSolver, sample_interior

HERE = Path(__file__).parent
CSV = HERE / "icnn_t12.csv"

SCHEDULES = ("lbfgs", "adam")
"""Table 2's rows: Adam then L-BFGS (the preset), and Adam alone."""

SUMMARY = ("loss", "train_rms", "test_rms", "test_max", "grid_max", "into", "onto")
"""The columns of the receipt that icnn.md cites."""

BOOTSTRAP = 10_000
"""Resamples of the seeds for the band of the mean, from numpy's generator seed 0."""

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


@dataclass(frozen=True, kw_only=True)
class SharedAngleSampler(TransportSampler):
    """Boundary points of a disk and target points of an ellipse at the same angles.

    A diagnostic sampler (icnn.md, T12, Finding 3) for a transport problem
    from a :class:`Disk` onto an :class:`Ellipse`: the interior points as
    :class:`TransportSampler` draws them, then ``boundary_points`` angles
    ``t``, uniform, giving the points ``center + r (cos t, sin t)`` on the
    circle and ``center + (a1 cos t, a2 sin t)`` on the ellipse. T12's exact
    map sends each boundary point to the target point of its angle, so its
    Hausdorff-type term is 0 here, not about 1e-4 as with independent
    points. ``target_points`` is ignored: there are as many target points
    as boundary points.
    """

    def supports(self, problem: Problem) -> bool:
        """Whether the problem maps a disk onto an ellipse."""
        boundary = problem.boundary
        return (
            isinstance(problem.domain, Disk)
            and isinstance(boundary, Transport)
            and isinstance(boundary.target, Ellipse)
        )

    def sample(self, problem: Problem, rng: np.random.Generator) -> Collocation:
        disk, boundary = problem.domain, problem.boundary
        assert isinstance(disk, Disk) and isinstance(boundary, Transport)
        assert isinstance(boundary.target, Ellipse)
        interior = sample_interior(disk, self.interior_points, rng)
        t = rng.uniform(0.0, 2.0 * np.pi, self.boundary_points)
        unit = np.stack([np.cos(t), np.sin(t)], axis=-1)
        return Collocation(
            interior=interior,
            boundary=np.asarray(disk.center) + disk.radius * unit,
            target_boundary=np.asarray(boundary.target.center)
            + np.asarray(boundary.target.semi_axes) * unit,
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


def run_seed(
    schedule: str, seed: int, options: dict, name: str, shared_angles: bool = False
) -> dict:
    """Train one seed under one schedule, log it to MLflow, return its row.

    With ``shared_angles`` the preset's sampler is replaced by
    :class:`SharedAngleSampler` with the same counts.
    """
    import mlflow
    import torch

    torch.set_num_threads(1)
    preset = solver(schedule, seed, options)
    solving: NeuralSolver = preset
    if shared_angles:
        sampler = SharedAngleSampler(
            interior_points=preset.interior_points,
            boundary_points=preset.boundary_points,
        )
        solving = with_parts(preset, sampler=sampler)
    logger = tracking.MLflowLogger()
    with tracking.run("maesolvers-icnn-T12", run_name=name.format(schedule, seed)):
        result = solving.solve(T12, callbacks=[logger])
        train = gradient_errors(result, solving.collocation(T12).interior)
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
    return f"{np.mean(values):.2e} ({np.std(values, ddof=1):.2e})"


def mean_band(values: list[float]) -> str:
    """Bootstrap 5% to 95% band of the mean over the seeds (``BOOTSTRAP`` resamples)."""
    rng = np.random.default_rng(0)
    draws = rng.choice(values, size=(BOOTSTRAP, len(values))).mean(axis=1)
    return f"[{np.quantile(draws, 0.05):.2e}, {np.quantile(draws, 0.95):.2e}]"


def summary(path: Path) -> None:
    """Print the cells of icnn.md's T12 tables from a receipt, three digits each."""
    with path.open() as file:
        rows = list(csv.DictReader(file))
    for schedule in sorted({r["schedule"] for r in rows}, key=SCHEDULES.index):
        mine = [r for r in rows if r["schedule"] == schedule]
        for key in SUMMARY:
            values = [float(r[key]) for r in mine]
            s = summarize_errors(values, threshold=np.inf)
            print(
                f"{schedule} {key}: mean (std) {mean_std(values)}; "
                f"median [5%, 95%] {s.median:.2e} [{s.q05:.2e}, {s.q95:.2e}]; "
                f"mean's band {mean_band(values)}"
            )


def exact_map_floor(seed: int, parametric: bool = False) -> tuple[float, float]:
    """The two halves of eq. (3.4) that T12's exact map leaves at a seed's points.

    The boundary and target points are the preset's collocation points of
    the seed; with ``parametric`` the target points are instead 800 points
    uniform in the parameter angle of the ellipse, from numpy's generator
    seed ``100 + seed``.
    """
    import torch

    from maesolvers.neural.functional import nearest_squared_distances

    collocation = TransportSampler().sample(T12, np.random.default_rng(seed))
    x = collocation.boundary
    image = np.stack([2.0 * x[:, 0] + 3.5, 0.5 * x[:, 1]], axis=-1)
    target = collocation.target_boundary
    if parametric:
        t = np.random.default_rng(100 + seed).uniform(0.0, 2.0 * np.pi, len(target))
        target = np.stack([3.5 + 2.0 * np.cos(t), 0.5 * np.sin(t)], axis=-1)
    into, onto = nearest_squared_distances(torch.tensor(image), torch.tensor(target))
    return float(into.mean()), float(onto.mean())


def floor(seeds: list[int]) -> None:
    """Print :func:`exact_map_floor` of each seed, both ways of drawing the target."""
    for seed in seeds:
        into, onto = exact_map_floor(seed)
        par = sum(exact_map_floor(seed, parametric=True))
        print(
            f"seed {seed}: into {into:.3e} + onto {onto:.3e} = {into + onto:.3e}; "
            f"parametric target {par:.3e}"
        )


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
    parser.add_argument(
        "--shared-angles",
        action="store_true",
        help="boundary and target points at shared angles (icnn.md, T12, Finding 3)",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--name", default="", help="prefix of the MLflow run names, e.g. 'diagnostic'"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--summary", action="store_true", help="print the tables' cells from the CSV"
    )
    mode.add_argument(
        "--floor", action="store_true", help="print the exact map's transport term"
    )
    args = parser.parse_args()
    if args.summary:
        summary(args.out or CSV)
        return
    if args.floor:
        floor(args.seeds)
        return
    options = {name: getattr(args, name) for name in OPTIONS}
    changed = {
        k: v
        for k, v in options.items()
        if k != "accelerator" and v != getattr(TransportICNN, k)
    }
    if args.shared_angles:
        changed["sampler"] = "shared-angles"
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
            pool.submit(run_seed, schedule, seed, options, name, args.shared_angles)
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
    summary(out)
    for schedule in args.schedules:
        stops = [r["stop"] for r in rows if r["schedule"] == schedule]
        print(
            f"{schedule} stops: {dict((s, stops.count(s)) for s in sorted(set(stops)))}"
        )
    print(f"wrote {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
