"""Regenerate comparison.csv: every preset of SOLVERS on the shared test set.

Runs each preset of ``maesolvers.SOLVERS`` with its defaults on each problem of
``SHARED_TEST_SET`` it supports: a classical preset once, on the grid
``n = 361``, the largest N of the papers (test-set specification, Section 2);
a neural preset once per seed, seeds 0 to 9 (ADR 0008). A pair the preset does
not support gets one row with status ``unsupported`` and the reason. Each row
holds :func:`~maesolvers.metrics.evaluate`, the run's convergence, the machine
and the preset's ``hparams``; the columns, statuses, grid, seeds and the
options each preset takes are those of comparison.py, which turns the CSV into
the README table.

Every run is one MLflow run in maesolvers-<preset>-<problem> (the preset's name
in lower case), named ``comparison`` and its grid or seed, tagged
``benchmark=comparison``. The runs take --workers processes (default 16), each
with one CPU thread; the neural presets train on the CPU, their default, and
the longest runs start first. With --solvers or --problems only those pairs
run, and their rows replace the same pairs' rows of the CSV given by --out, so
one cell can be rerun. Run it on a100server from the repository root, writing
outside the checkout so that the runs do not see it dirty, then copy the CSV
back::

    just remote-bg comparison python benchmarks/run_comparison.py \\
        --out /jczhang02/maesolvers-data/data/comparison.csv

A run with --n, --max-steps or --seeds (a smoke test, say) differs from the
table's and needs --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from comparison import (
    CSV,
    FIELDS,
    SEEDS,
    N,
    merge,
    preset_options,
    read,
    unsupported_reason,
)

from maesolvers import CATALOGUE, SOLVERS, NeuralSolver, tracking
from maesolvers.catalogue import SHARED_TEST_SET
from maesolvers.metrics import evaluate, seed_error

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def is_neural(name: str) -> bool:
    """Whether the preset ``name`` is a neural solver, trained once per seed."""
    return issubclass(SOLVERS[name], NeuralSolver)


def commit() -> str:
    """The checkout's commit, with ``+dirty`` if tracked files have changes."""
    tags = tracking.git_tags()
    dirty = "+dirty" if tags["git.dirty"] == "true" else ""
    return tags["mlflow.source.git.commit"] + dirty


def cpu() -> str:
    """The CPU's model name, from /proc/cpuinfo on Linux."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return " ".join(line.split(":", 1)[1].split())
    return platform.processor() or platform.machine()


def csv_field(value: Any) -> Any:
    """A CSV field: floats in scientific notation with seven digits."""
    return f"{value:.6e}" if isinstance(value, float) else value


def run_one(task: tuple[str, str, int | None, dict[str, Any], bool]) -> dict[str, Any]:
    """Run one preset on one problem (one seed), return its row of comparison.csv.

    The task is the preset's name, the problem's T id, the seed (None for a
    classical preset), the runner's options and whether to log to MLflow; the
    preset takes those of the options and the seed its constructor has.
    """
    name, problem_id, seed, options, log = task
    problem = CATALOGUE[problem_id]
    kwargs = preset_options(name, {**options, "seed": seed})
    solver = SOLVERS[name](**kwargs)
    row: dict[str, Any] = {
        "solver": name,
        "problem": problem_id,
        "seed": "" if seed is None else seed,
        "n": kwargs.get("n", ""),
        "threads": 1,
        "cpu": cpu(),
        "commit": commit(),
        "hparams": json.dumps(solver.hparams, sort_keys=True),
    }
    if seed is not None:
        import torch

        torch.set_num_threads(1)
    label = " ".join(f"{k}={v}" for k, v in kwargs.items())
    try:
        if log:
            logger = tracking.MLflowLogger()
            experiment = f"maesolvers-{name.lower()}-{problem_id}"
            run = tracking.run(
                experiment,
                run_name=f"comparison {label}",
                tags={"benchmark": "comparison"},
            )
            with run:
                result = solver.solve(problem, callbacks=[logger])
            metrics = logger.evaluation or evaluate(result)
        else:
            result = solver.solve(problem)
            metrics = evaluate(result)
    except Exception as error:  # a failed run is a row, never dropped (ADR 0007)
        message = " ".join(f"{type(error).__name__}: {error}".split())
        print(f"{name} {problem_id} {label}: failed, {message}", flush=True)
        return {**row, "status": "failed", "reason": message[:200]}
    info = result.info
    stop = info.get("stop", "")
    if seed_error(result) is None:  # ADR 0007, as summarize counts it
        finite = np.isfinite(metrics.get("max", np.nan))
        status, reason = "failed", f"stopped as {stop}" if finite else "non-finite"
    else:
        status, reason = "ok", ""
    row.update(
        {key: metrics[key] for key in FIELDS if key in metrics},
        status=status,
        reason=reason,
        converged=bool(info.get("converged", False)),
        stop=stop,
        device=info.get("device", "cpu"),
    )
    print(f"{name} {problem_id} {label}: max {row.get('max', 'n/a')}", flush=True)
    return row


def tasks(
    solvers: list[str],
    problems: list[str],
    seeds: list[int],
    options: dict[str, Any],
    log: bool,
) -> tuple[list[tuple[str, str, int | None, dict[str, Any], bool]], list[dict]]:
    """The runs, longest first, and the rows of the unsupported pairs."""
    runs, unsupported = [], []
    for problem_id in problems:
        problem = CATALOGUE[problem_id]
        for name in solvers:
            solver = SOLVERS[name](**preset_options(name, options))
            if not solver.supports(problem):
                unsupported.append(
                    {
                        "solver": name,
                        "problem": problem_id,
                        "status": "unsupported",
                        "reason": unsupported_reason(solver, problem),
                    }
                )
                continue
            for seed in seeds if is_neural(name) else (None,):
                runs.append((name, problem_id, seed, options, log))
    # Neural runs take minutes, the higher dimensions most; classical ones less.
    runs.sort(key=lambda t: (not is_neural(t[0]), -CATALOGUE[t[1]].domain.dim))
    return runs, unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solvers", nargs="+", choices=list(SOLVERS), default=None)
    parser.add_argument(
        "--problems", nargs="+", choices=SHARED_TEST_SET, default=list(SHARED_TEST_SET)
    )
    parser.add_argument("--n", type=int, default=N, help="classical grid")
    parser.add_argument("--max-steps", type=int, default=None, help="neural budget")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--out", type=Path, default=CSV)
    args = parser.parse_args()
    options: dict[str, Any] = {"n": args.n, "max_steps": args.max_steps}
    changed = args.n != N or args.max_steps is not None or args.seeds != list(SEEDS)
    if changed and args.out == CSV:
        parser.error("--n, --max-steps or --seeds differ from the table's: give --out")
    solvers = args.solvers or list(SOLVERS)
    log = not args.no_mlflow
    runs, unsupported = tasks(solvers, args.problems, args.seeds, options, log)
    for variable in THREAD_VARIABLES:  # inherited by the spawned workers
        os.environ[variable] = "1"
    start = time.time()
    context = get_context("spawn")  # CUDA cannot be forked
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        rows = list(pool.map(run_one, runs))
    old = read(args.out) if args.out.exists() else []
    rows = merge(old, [*unsupported, *rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: csv_field(v) for k, v in row.items()} for row in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    print(f"{len(runs)} runs, {failed} failed rows in the CSV")
    print(f"wrote {args.out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
