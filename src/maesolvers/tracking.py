"""Log experiments to the project's MLflow server.

Use one run per experiment configuration::

    from maesolvers import tracking

    with tracking.run("maesolvers-icnn-T1", params={"seed": 0, "steps": 2000}):
        ...
        mlflow.log_metric("max_grid", err)

`run` tags each run with the commit and branch of the checkout that holds
this package, and `git.dirty`, so a result from uncommitted code is
visible. It does not rely on MLflow's own git detection, which only works
when the entry script is inside the repo.

The server URI comes from MLFLOW_TRACKING_URI and defaults to the shared
MLflow server on a100server (see justfile, mlflow-ui). mlflow is in the
`experiments` dependency group, not a library dependency.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow

DEFAULT_URI = "http://127.0.0.1:30000"
REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_tags() -> dict[str, str]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "mlflow.source.git.commit": _git("rev-parse", "HEAD"),
        "git.branch": _git("branch", "--show-current"),
        "git.dirty": "unknown" if status == "unknown" else str(bool(status)).lower(),
    }


@contextmanager
def run(experiment: str, params: dict | None = None, **kwargs):
    """Start an MLflow run in `experiment` and log `params`; yield the run."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(**kwargs) as active:
        mlflow.set_tags(_git_tags())
        if params:
            mlflow.log_params(params)
        yield active
