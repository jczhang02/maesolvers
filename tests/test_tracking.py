"""MLflow logging: the git tags, the flattened parameters and the logger."""

import subprocess
import sys
import types
from contextlib import contextmanager

import pytest

from maesolvers import T1, T2, FDNewton, Newton, Standard9, tracking

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
        self.entities = types.ModuleType("mlflow.entities")
        self.entities.Metric = lambda key, value, timestamp, step: (key, value, step)
        fake = self

        class MlflowClient:
            def log_batch(self, run_id, metrics=()):
                fake.calls.append(("batch", run_id, list(metrics)))

        self.MlflowClient = MlflowClient

    def install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", self)
        monkeypatch.setitem(sys.modules, "mlflow.entities", self.entities)
        return self

    def active_run(self):
        if not self.active:
            return None
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="id"))

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
    fake = FakeMLflow().install(monkeypatch)
    logger = tracking.MLflowLogger()
    result = FDNewton(n=8).solve(T1, callbacks=[logger])
    tags, params, batch, final = fake.calls
    assert tags == ("tags", ["git.branch", "git.dirty", "mlflow.source.git.commit"])
    assert params == (
        "params",
        {
            "class_path": "maesolvers.classical.presets.FDNewton",
            "max_iter": 50,
            "tol": 1e-10,
            "n": 8,
            "spline_degree": 3,
            "problem": "T1",
        },
    )
    history = result.info["history"]
    assert batch == (
        "batch",
        "id",
        [("residual", r, i) for i, r in enumerate(history, start=1)],
    )
    kind, metrics, step = final
    assert kind == "metrics" and step is None
    assert metrics["converged"] == 1.0 and metrics["iterations"] == len(history)
    assert {"max", "rms", "max_grid", "eigmin", "time", "unknowns"} <= set(metrics)
    assert all(type(v) is float for v in metrics.values())
    assert logger.evaluation["max"] == metrics["max"]


def test_logger_sends_nothing_between_start_and_end(monkeypatch):
    fake = FakeMLflow().install(monkeypatch)
    solver, logger = FDNewton(n=8), tracking.MLflowLogger()
    logger.on_solve_start(solver, T1)
    logged = list(fake.calls)
    for step in (1, 2, 3):
        logger.on_step_end(solver, step, {"residual": 1.0 / step})
    assert fake.calls == logged
    logger.on_solve_end(solver, solver.solve(T1))
    batches = [call for call in fake.calls if call[0] == "batch"]
    assert batches == [
        (
            "batch",
            "id",
            [("residual", 1.0, 1), ("residual", 0.5, 2), ("residual", 1 / 3, 3)],
        )
    ]


def test_logger_needs_an_active_run(monkeypatch):
    FakeMLflow(active=False).install(monkeypatch)
    with pytest.raises(RuntimeError, match="tracking.run"):
        FDNewton(n=8).solve(T1, callbacks=[tracking.MLflowLogger()])


def test_run_sets_the_experiment_tags_and_params(monkeypatch):
    fake = FakeMLflow().install(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/nowhere")
    with tracking.run("maesolvers-x-T1", params={"seed": 0}, run_name="r") as active:
        assert active == "run"
    assert fake.calls == [
        ("uri", "file:/nowhere"),
        ("experiment", "maesolvers-x-T1"),
        ("start", {"run_name": "r"}),
        ("tags", ["git.branch", "git.dirty", "mlflow.source.git.commit"]),
        ("params", {"seed": 0}),
        ("end",),
    ]


def test_logger_against_a_real_mlflow_store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    with tracking.run("maesolvers-fdnewton-T2") as active:
        FDNewton(n=8).solve(T2, callbacks=[tracking.MLflowLogger()])
    data = mlflow.get_run(active.info.run_id).data
    assert data.params["n"] == "8" and data.params["problem"] == "T2"
    assert data.metrics["converged"] == 1.0 and "max" in data.metrics
    assert "git.dirty" in data.tags
    history = mlflow.MlflowClient().get_metric_history(active.info.run_id, "residual")
    assert [m.step for m in history] == list(range(1, len(history) + 1))


def test_importing_the_library_loads_no_extra():
    code = (
        "import sys, maesolvers, maesolvers.tracking; "
        "loaded = {'mlflow', 'torch', 'jsonargparse'} & set(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
