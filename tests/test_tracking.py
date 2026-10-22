"""MLflow logging: the git tags, the flattened parameters and the logger."""

import subprocess
import sys
import types
from contextlib import contextmanager

import pytest

from maesolvers import P4, P5, FDNewton, Newton, Standard9, tracking

_ISOLATED = (
    *("-c", "user.name=t", "-c", "user.email=t@t"),
    *("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"),
)


def _git(cwd, *args):
    subprocess.run(
        ["git", *_ISOLATED, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_git_tags_come_from_the_working_directory(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "trunk")
    (tmp_path / "a.txt").write_text("1")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)
    assert tracking.git_tags() == {
        "mlflow.source.git.commit": head,
        "git.branch": "trunk",
        "git.dirty": "false",
    }
    (tmp_path / "a.txt").write_text("2")
    assert tracking.git_tags()["git.dirty"] == "true"


def test_git_tags_outside_a_checkout_are_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert set(tracking.git_tags().values()) == {"unknown"}


def test_flatten_dots_nested_parts():
    config = Newton(scheme=Standard9(n=8), max_iter=3).hparams
    assert tracking.flatten(config) == {
        "class_path": "maesolvers.classical.iterations.Newton",
        "scheme.class_path": "maesolvers.classical.schemes.Standard9",
        "scheme.n": 8,
        "scheme.spline_degree": 3,
        "max_iter": 3,
        "tol": 1e-10,
    }
    assert tracking.flatten({"terms": ({"w": 1}, 2)}) == {"terms.0.w": 1, "terms.1": 2}


class FakeMLflow(types.ModuleType):
    """Records the calls the tracking module makes."""

    def __init__(self, active=True):
        super().__init__("mlflow")
        self.active, self.calls = active, []

    def active_run(self):
        return object() if self.active else None

    def set_tracking_uri(self, uri):
        self.calls.append(("uri", uri))

    def set_experiment(self, name):
        self.calls.append(("experiment", name))

    @contextmanager
    def start_run(self, **kwargs):
        self.calls.append(("start", kwargs))
        yield "run"
        self.calls.append(("end",))

    def set_tags(self, tags):
        self.calls.append(("tags", sorted(tags)))

    def log_params(self, params):
        self.calls.append(("params", params))

    def log_metrics(self, metrics, step=None):
        self.calls.append(("metrics", metrics, step))


def test_logger_logs_hparams_steps_and_the_evaluation(monkeypatch):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    result = FDNewton(n=8).solve(P4, callbacks=[tracking.MLflowLogger()])
    tags, params, *steps, final = fake.calls
    assert tags == ("tags", ["git.branch", "git.dirty", "mlflow.source.git.commit"])
    assert params == (
        "params",
        {
            "class_path": "maesolvers.classical.presets.FDNewton",
            "max_iter": 50,
            "tol": 1e-10,
            "n": 8,
            "spline_degree": 3,
            "problem": "P4",
        },
    )
    history = result.info["history"]
    assert steps == [
        ("metrics", {"residual": r}, i) for i, r in enumerate(history, start=1)
    ]
    kind, metrics, step = final
    assert kind == "metrics" and step is None
    assert metrics["converged"] == 1.0 and metrics["iterations"] == len(history)
    assert {"max", "rms", "max_grid", "eigmin", "time", "unknowns"} <= set(metrics)
    assert all(type(v) is float for v in metrics.values())


def test_logger_needs_an_active_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", FakeMLflow(active=False))
    with pytest.raises(RuntimeError, match="tracking.run"):
        FDNewton(n=8).solve(P4, callbacks=[tracking.MLflowLogger()])


def test_run_sets_the_experiment_tags_and_params(monkeypatch):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/nowhere")
    with tracking.run("maesolvers-x-P4", params={"seed": 0}, run_name="r") as active:
        assert active == "run"
    assert fake.calls == [
        ("uri", "file:/nowhere"),
        ("experiment", "maesolvers-x-P4"),
        ("start", {"run_name": "r"}),
        ("tags", ["git.branch", "git.dirty", "mlflow.source.git.commit"]),
        ("params", {"seed": 0}),
        ("end",),
    ]


def test_logger_against_a_real_mlflow_store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    with tracking.run("maesolvers-fdnewton-P5") as active:
        FDNewton(n=8).solve(P5, callbacks=[tracking.MLflowLogger()])
    data = mlflow.get_run(active.info.run_id).data
    assert data.params["n"] == "8" and data.params["problem"] == "P5"
    assert data.metrics["converged"] == 1.0 and "max" in data.metrics
    assert "git.dirty" in data.tags
    history = mlflow.MlflowClient().get_metric_history(active.info.run_id, "residual")
    assert [m.step for m in history] == list(range(1, len(history) + 1))
