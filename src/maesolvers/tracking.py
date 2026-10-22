"""Log experiments to the project's MLflow server.

Open one run per configuration with :func:`run` and pass an
:class:`MLflowLogger` to the solve, which logs the solver's hyperparameters,
every step's metrics and the final :func:`~maesolvers.metrics.evaluate`
into that run::

    from maesolvers import T1, FDPoisson, tracking

    with tracking.run("maesolvers-fdpoisson-T1"):
        FDPoisson(n=64).solve(T1, callbacks=[tracking.MLflowLogger()])

Both tag the run with the commit and branch of the git checkout that holds
the current working directory, and ``git.dirty``, so a result from
uncommitted code is visible. Run experiments from the checkout whose code
they use. MLflow's own git detection is not used: it only works when the
entry script is inside the repo.

The server URI comes from ``MLFLOW_TRACKING_URI`` and defaults to the shared
MLflow server on a100server (see justfile, mlflow-ui). mlflow is in the
``experiments`` dependency group, not a library dependency; it is imported
only when a run starts or a logger is used.
"""

from __future__ import annotations

import numbers
import os
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from maesolvers.core.callbacks import Callback
from maesolvers.core.problem import Problem
from maesolvers.core.result import Result
from maesolvers.metrics import evaluate

if TYPE_CHECKING:
    from mlflow import ActiveRun

    from maesolvers.core.solver import Solver

DEFAULT_URI = "http://127.0.0.1:30000"


def _git(*args: str) -> str:
    """Output of a git command in the current working directory, or "unknown"."""
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_tags() -> dict[str, str]:
    """Read the git state of the checkout that holds the current working directory.

    Returns
    -------
    dict
        ``mlflow.source.git.commit``, ``git.branch`` and ``git.dirty``
        (``"true"`` if tracked files have uncommitted changes); ``"unknown"``
        for each outside a checkout.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "mlflow.source.git.commit": _git("rev-parse", "HEAD"),
        "git.branch": _git("branch", "--show-current"),
        "git.dirty": "unknown" if status == "unknown" else str(bool(status)).lower(),
    }


@contextmanager
def run(
    experiment: str, params: Mapping[str, Any] | None = None, **kwargs: Any
) -> Iterator[ActiveRun]:
    """Start an MLflow run in ``experiment``, tagged with :func:`git_tags`.

    Parameters
    ----------
    experiment
        Experiment name, ``maesolvers-<method>-<problem>`` on the shared server.
    params
        Parameters to log at the start, or None.
    **kwargs
        Passed to ``mlflow.start_run``, such as ``run_name``.

    Yields
    ------
    mlflow.ActiveRun
    """
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(**kwargs) as active:
        mlflow.set_tags(git_tags())
        if params:
            mlflow.log_params(dict(params))
        yield active


def flatten(config: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a configuration into MLflow parameters with dotted keys.

    Parameters
    ----------
    config
        A configuration as in :attr:`Solver.hparams
        <maesolvers.core.solver.Solver.hparams>`: ``class_path`` and
        ``init_args``, nested parts, containers and plain values.
    prefix
        Prepended to every key.

    Returns
    -------
    dict
        For ``Newton(scheme=Standard9(n=8))``: ``class_path``,
        ``scheme.class_path``, ``scheme.n``, ``scheme.spline_degree``,
        ``max_iter`` and ``tol``. Container elements are keyed by index.
    """
    if isinstance(config, Mapping) and set(config) == {"class_path", "init_args"}:
        out = {f"{prefix}class_path": config["class_path"]}
        out.update(flatten(config["init_args"], prefix))
        return out
    if isinstance(config, Mapping):
        items = [(str(k), v) for k, v in config.items()]
    elif isinstance(config, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(config)]
    else:
        return {prefix[:-1]: config}
    out = {}
    for key, value in items:
        out.update(flatten(value, f"{prefix}{key}."))
    return out


class MLflowLogger(Callback):
    """Log a solve into the active MLflow run.

    At the start it sets :func:`git_tags` and logs the solver's flattened
    :attr:`~maesolvers.core.solver.Solver.hparams` (:func:`flatten`) and the
    problem's name as parameters. It keeps every step's metrics (``residual``,
    ``update``, ``loss``) and sends them at the end in one batch, together
    with the numeric fields of :func:`~maesolvers.metrics.evaluate` and
    ``converged`` (1 or 0). So no MLflow call falls between the start and the
    end of the solve, the region whose time ``Result.info["time"]`` reports.
    Use one run per solve, opened with :func:`run`: MLflow refuses to change
    a logged parameter.

    Attributes
    ----------
    evaluation : dict or None
        :func:`~maesolvers.metrics.evaluate` of the last result logged, None
        before the first solve ends.

    Raises
    ------
    RuntimeError
        At the start of a solve, if no MLflow run is active.
    """

    def __init__(self) -> None:
        self.evaluation: dict[str, Any] | None = None
        self._steps: list[tuple[str, float, int, int]] = []

    def on_solve_start(self, solver: Solver, problem: Problem) -> None:
        """Tag the run and log the hyperparameters and the problem."""
        import mlflow

        if mlflow.active_run() is None:
            raise RuntimeError(
                "MLflowLogger logs into the active MLflow run; open one with "
                "maesolvers.tracking.run(experiment)"
            )
        self._steps = []
        mlflow.set_tags(git_tags())
        mlflow.log_params({**flatten(solver.hparams), "problem": problem.name})

    def on_step_end(
        self, solver: Solver, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Keep the step's metrics, with the time and step, for the end."""
        timestamp = int(time.time() * 1000)
        self._steps.extend(
            (key, float(value), timestamp, step) for key, value in metrics.items()
        )

    def on_solve_end(self, solver: Solver, result: Result) -> None:
        """Send the kept step metrics, the evaluation and whether it converged."""
        import mlflow
        from mlflow.entities import Metric

        if self._steps:
            mlflow.MlflowClient().log_batch(
                mlflow.active_run().info.run_id,
                metrics=[Metric(*step) for step in self._steps],
            )
            self._steps = []
        self.evaluation = evaluate(result)
        metrics = {
            key: float(value)
            for key, value in self.evaluation.items()
            if isinstance(value, numbers.Real) and not isinstance(value, bool)
        }
        metrics["converged"] = float(bool(result.info.get("converged", False)))
        mlflow.log_metrics(metrics)
