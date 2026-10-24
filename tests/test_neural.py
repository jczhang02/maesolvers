"""Training a neural solver on CPU: the smoke tests of the neural extra.

Every solve here trains for a handful of steps on a few points, so the
tests check the protocol (ADR 0003, 0008), not the accuracy; the accuracy
is the reproduction's business (benchmarks/penalty_mlp.md).
"""

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from maesolvers import (  # noqa: E402
    LBFGS,
    MLP,
    T1,
    T6,
    T8,
    T16,
    Adam,
    Callback,
    HardDirichlet,
    NeuralSolver,
    PenaltyLoss,
    PenaltyMLP,
    UniformSampler,
    tracking,
)
from maesolvers.metrics import evaluate, seed_error  # noqa: E402
from maesolvers.neural import (  # noqa: E402
    Batch,
    Collocation,
    Loss,
    Network,
    Optimizer,
)
from maesolvers.neural import functional as F  # noqa: E402

pytestmark = pytest.mark.neural

SMALL = {"interior_points": 60, "boundary_points": 24, "accelerator": "cpu"}


def test_a_cpu_solve_on_a_gpu_host_does_not_advise_the_gpu(monkeypatch):
    from lightning.pytorch.accelerators import CUDAAccelerator

    monkeypatch.setattr(CUDAAccelerator, "is_available", staticmethod(lambda: True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PenaltyMLP(max_steps=2, **SMALL).solve(T1)
    assert not [w for w in caught if "GPU available" in str(w.message)]


class Recorder(Callback):
    def __init__(self):
        self.events = []

    def on_solve_start(self, solver, problem):
        self.events.append(("start", problem.name))

    def on_step_end(self, solver, step, metrics):
        self.events.append(("step", step, dict(metrics)))

    def on_solve_end(self, solver, result):
        self.events.append(("end", result.info["iterations"]))


@pytest.fixture(scope="module")
def t6_result():
    return PenaltyMLP(max_steps=4, **SMALL).solve(T6)


def test_a_result_is_a_callable_network_with_an_autograd_hessian(t6_result):
    result = t6_result
    points = np.array([[[0.2, 0.3], [0.1, 0.6]], [[0.4, 0.2], [0.3, 0.3]]])
    values = result.u(points)
    assert values.shape == (2, 2) and values.dtype == np.float64
    assert result.u([0.2, 0.3]).shape == ()
    hess = result.hessian(points)
    assert hess.shape == (2, 2, 2, 2) and hess.dtype == np.float64
    np.testing.assert_allclose(hess, np.swapaxes(hess, -1, -2), atol=1e-10)
    assert result.nodes is None and result.node_hessian is None
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        result.u(np.zeros((3, 3)))


def test_the_hessian_agrees_with_finite_differences_of_u(t6_result):
    x = np.array([0.3, 0.4])
    h = 1e-4
    e = np.eye(2) * h
    u = t6_result.u
    fd = np.array(
        [
            [
                (
                    u(x + e[i] + e[j])
                    - u(x + e[i] - e[j])
                    - u(x - e[i] + e[j])
                    + u(x - e[i] - e[j])
                )
                / (4 * h * h)
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(t6_result.hessian(x), fd, rtol=1e-4, atol=1e-4)


def test_evaluate_measures_a_neural_result_on_the_shared_grid(t6_result):
    metrics = evaluate(t6_result)
    assert {"max", "rms", "nonconvex_fraction", "eigmin", "eigmin_exact"} <= set(
        metrics
    )
    assert "max_grid" not in metrics  # no nodes: the shared grid is primary
    assert metrics["unknowns"] == 481 and metrics["iterations"] == 4
    assert np.isfinite(metrics["max"])


def test_the_result_records_how_the_training_went(t6_result):
    info = t6_result.info
    assert info["stop"] == "max_steps" and info["converged"] is False
    assert len(info["history"]) == info["iterations"] == 4
    assert info["device"] == "cpu" and info["threads"] == torch.get_num_threads()
    assert {"loss", "equation", "convexity"} == set(info["loss_terms"])
    assert info["collocation"] == {
        "interior_points": 60,
        "boundary_points": 24,
        "target_boundary_points": 0,
        "fingerprint": PenaltyMLP(**SMALL).collocation(T6).fingerprint(),
    }
    assert t6_result.hparams["init_args"]["max_steps"] == 4


def test_callbacks_see_every_step_with_the_loss_terms():
    recorder = Recorder()
    result = PenaltyMLP(max_steps=3, **SMALL).solve(T1, callbacks=[recorder])
    assert recorder.events[0] == ("start", "T1")
    steps = recorder.events[1:-1]
    assert [event[1] for event in steps] == [1, 2, 3]
    assert all(set(event[2]) == {"loss", "equation", "convexity"} for event in steps)
    assert [event[2]["loss"] for event in steps] == list(result.info["history"])
    assert recorder.events[-1] == ("end", 3)


def test_the_loss_decreases_under_lbfgs():
    history = PenaltyMLP(max_steps=10, **SMALL).solve(T1).info["history"]
    assert history[-1] < history[0]
    assert all(b <= a for a, b in itertools.pairwise(history))


def test_the_lbfgs_part_makes_one_iteration_per_step():
    # torch.optim.LBFGS makes up to max_iter = 20 iterations per step() by
    # default; the training loop counts one update per step
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0], dtype=torch.float64))
    optimizer = LBFGS().build([x])

    def closure():
        optimizer.zero_grad()
        loss = x[0] ** 2 + 10 * x[1] ** 2
        loss.backward()
        return loss

    for _ in range(3):
        optimizer.step(closure)
    assert optimizer.state[x]["n_iter"] == 3


def test_a_seed_fixes_the_result_and_leaves_torch_as_it_was():
    torch.manual_seed(123)
    state, dtype = torch.get_rng_state(), torch.get_default_dtype()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    first = PenaltyMLP(max_steps=2, **SMALL).solve(T1)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.get_default_dtype() == dtype
    if torch.cuda.is_available():
        after = torch.cuda.get_rng_state_all()
        assert all(torch.equal(a, b) for a, b in zip(cuda, after, strict=True))
    again = PenaltyMLP(max_steps=2, **SMALL).solve(T1)
    other = PenaltyMLP(max_steps=2, seed=1, **SMALL).solve(T1)
    points = np.array([[0.3, 0.6], [0.5, 0.5]])
    np.testing.assert_array_equal(first.u(points), again.u(points))
    assert not np.allclose(first.u(points), other.u(points))


def test_the_hard_ansatz_reports_its_fits_and_both_fits_converge():
    solver = PenaltyMLP(max_steps=1, **SMALL)
    report = solver.solve(T1).info["network"]
    for fit in ("lift", "distance"):
        assert report[fit]["converged"], report[fit]
        assert report[fit]["max_gradient"] <= solver.fit_tol
        assert 0 < report[fit]["iterations"] < solver.fit_steps
    # The fits stop on the gradient of a mean square, not on the misfit: at
    # fit_tol = 1e-5, G misses g and D misses 0 by about 1e-2 on the boundary
    # (benchmarks/penalty_mlp.md, Finding 2).
    assert report["lift"]["max_misfit"] == report["lift_boundary_misfit"] < 1e-2
    assert report["distance_boundary_max"] < 1.5e-2


def test_a_fit_that_hits_its_cap_reports_it():
    solver = PenaltyMLP(max_steps=1, fit_steps=3, **SMALL)
    report = solver.solve(T1).info["network"]
    assert report["distance"]["iterations"] == 3
    assert not report["distance"]["converged"]


def test_a_large_tolerance_converges_before_the_first_step():
    result = PenaltyMLP(max_steps=5, tol=1e12, **SMALL).solve(T1)
    assert result.info["converged"] and result.info["stop"] == "converged"
    assert result.info["iterations"] == 0 and result.info["history"] == ()


@dataclass(frozen=True, kw_only=True)
class NotANumber(Loss):
    """A loss of your own: here one that is NaN."""

    def __call__(self, u, batch):
        value = u(batch.interior).sum() * float("nan")
        return {"loss": value}


@dataclass(frozen=True, kw_only=True)
class Frozen(Optimizer):
    """An optimizer of your own: here one whose steps change nothing."""

    def build(self, parameters):
        return torch.optim.SGD(parameters, lr=0.0)


def _composed(**kwargs):
    parts = {
        "network": MLP(width=4, depth=2),
        "loss": PenaltyLoss(boundary_weight=10.0),
        "sampler": UniformSampler(interior_points=30, boundary_points=12),
        "optimizer": Adam(lr=1e-2),
        "max_steps": 3,
        "accelerator": "cpu",
    }
    return NeuralSolver(**{**parts, **kwargs})


def test_a_non_finite_loss_stops_the_training_as_diverged():
    result = _composed(loss=NotANumber()).solve(T1)
    assert result.info["stop"] == "diverged" and not result.info["converged"]
    assert result.info["iterations"] == 0
    assert seed_error(result) is None  # a failed seed for summarize (ADR 0007)


def test_a_step_that_changes_nothing_stops_the_training_as_stalled():
    recorder = Recorder()
    result = _composed(optimizer=Frozen()).solve(T1, callbacks=[recorder])
    assert result.info["stop"] == "stalled" and not result.info["converged"]
    assert result.info["iterations"] == 0 and result.info["history"] == ()
    assert [event[0] for event in recorder.events] == ["start", "end"]


def test_a_plain_network_trains_although_its_output_bias_gets_no_gradient():
    # The loss sees only the Hessian of an MLP, never its output bias.
    for optimizer in (LBFGS(), Adam(lr=1e-2)):
        result = _composed(
            loss=PenaltyLoss(),
            sampler=UniformSampler(interior_points=30, boundary_points=0),
            optimizer=optimizer,
        ).solve(T1)
        assert result.info["iterations"] == 3 and result.info["stop"] == "max_steps"
        assert "network" not in result.info  # an MLP fits nothing before training


def test_a_boundary_penalty_without_boundary_points_is_refused():
    sampler = UniformSampler(interior_points=30, boundary_points=0)
    solver = _composed(sampler=sampler)  # PenaltyLoss(boundary_weight=10.0)
    with pytest.raises(ValueError, match="sampler gave no boundary points"):
        solver.solve(T1)


def test_a_composed_solver_with_a_boundary_penalty_and_adam_trains():
    result = _composed(max_steps=5).solve(T8)
    assert result.info["iterations"] == 5
    assert set(result.info["loss_terms"]) == {
        "loss",
        "equation",
        "convexity",
        "boundary",
    }
    assert result.info["unknowns"] == (2 * 4 + 4) + (4 * 4 + 4) + (4 + 1)


@dataclass(frozen=True, kw_only=True)
class Quadratic(Network):
    """A network of your own: ``a |x|^2 + b`` with two trainable numbers."""

    def build(self, problem, collocation):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(0.1))
                self.b = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x):
                return self.a * (x**2).sum(dim=-1) + self.b

        return Module()


def test_a_network_of_your_own_plugs_in_and_trains_in_float64():
    # T8: u = (|x|^2 - 1) / 2, det D^2 u = 1, so a -> 1/2 and b -> -1/2.
    solver = _composed(
        network=Quadratic(),
        loss=PenaltyLoss(boundary_weight=1.0),
        optimizer=Adam(lr=0.05),
        max_steps=400,
        tol=1e-8,
    )
    result = solver.solve(T8)
    assert result.info["unknowns"] == 2
    assert result.u(np.zeros(2)) == pytest.approx(-0.5, abs=1e-2)
    np.testing.assert_allclose(result.hessian(np.zeros(2)), np.eye(2), atol=2e-2)


def test_the_hard_ansatz_trains_only_the_inner_network():
    collocation = PenaltyMLP(**SMALL).collocation(T1)
    with torch.random.fork_rng():
        torch.set_default_dtype(torch.float64)
        try:
            module = HardDirichlet(network=MLP(), fit_steps=50).build(T1, collocation)
        finally:
            torch.set_default_dtype(torch.float32)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert trainable == 481
    assert sum(p.numel() for p in module.parameters()) == 481 + 81 + 161


def test_the_hard_ansatz_needs_boundary_points():
    points = Collocation(interior=np.full((3, 2), 0.5), boundary=np.zeros((0, 2)))
    with pytest.raises(ValueError, match="HardDirichlet fits G and D"):
        HardDirichlet(network=MLP()).build(T1, points)


def test_the_loss_terms_of_the_penalty_loss():
    loss = PenaltyLoss(convexity_weight=2.0, boundary_weight=3.0)
    points = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float64)
    batch = type("B", (), {})()
    batch.interior, batch.boundary = points, points
    batch.f = torch.tensor([1.0, 3.0], dtype=torch.float64)
    batch.g = torch.zeros(2, dtype=torch.float64)

    def u(x):  # det D^2 u = 4 - 0 at every point; u_yy = -2 < 0
        return x[:, 0] ** 2 - x[:, 1] ** 2 + 1.0

    terms = loss(u, batch)
    hess = F.hessian(u, points)
    assert terms["equation"].item() == pytest.approx((-4 - 1) ** 2 + (-4 - 3) ** 2)
    assert terms["convexity"].item() == pytest.approx(4.0)
    assert terms["boundary"].item() == pytest.approx(
        F.mean_squared_misfit(u(points), batch.g).item()
    )
    expected = terms["equation"] + 2.0 * terms["convexity"] + 3.0 * terms["boundary"]
    assert terms["loss"].item() == pytest.approx(expected.item())
    assert hess.shape == (2, 2, 2)


def test_the_penalty_loss_refuses_a_batch_without_dirichlet_data():
    points = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    batch = Batch(
        interior=points,
        f=torch.ones(1),
        boundary=points,
        g=None,
        f_boundary=torch.ones(1),
    )
    with pytest.raises(ValueError, match="no Dirichlet data"):
        PenaltyLoss(boundary_weight=1.0)(lambda x: x.sum(dim=-1), batch)


@dataclass(frozen=True, kw_only=True)
class Residual(Loss):
    """A loss of your own for any boundary condition: the residual alone."""

    def supports(self, problem):
        return True

    def __call__(self, u, batch):
        if batch.g is not None:
            raise AssertionError("a transport problem has no Dirichlet data")
        if batch.target is not T16.boundary.target:
            raise AssertionError("a transport problem passes its target")
        hess = F.hessian(u, batch.interior)
        return {"loss": F.monge_ampere_residual(hess, batch.f).pow(2).mean()}


def test_a_batch_has_no_dirichlet_data_for_another_boundary_condition():
    solver = NeuralSolver(
        network=MLP(width=4, depth=1),
        loss=Residual(),
        sampler=UniformSampler(interior_points=20, boundary_points=8),
        optimizer=Adam(),
        max_steps=2,
    )
    result = solver.solve(T16)  # Residual checks batch.g and batch.target
    assert result.info["iterations"] == 2


def test_mlflow_logs_a_neural_solve(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    with tracking.run("maesolvers-penalty-mlp-T1") as active:
        PenaltyMLP(max_steps=3, **SMALL).solve(T1, callbacks=[tracking.MLflowLogger()])
    data = mlflow.get_run(active.info.run_id).data
    assert data.params["width"] == "10" and data.params["problem"] == "T1"
    assert "max" in data.metrics and data.metrics["converged"] == 0.0
    history = mlflow.MlflowClient().get_metric_history(active.info.run_id, "loss")
    assert [m.step for m in history] == [1, 2, 3]
