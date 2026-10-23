"""The convex PINN of Caboussat-Peruso 2025 and its parts (needs the neural extra).

The loss is checked on T12's exact solution, at which every term vanishes;
training runs a handful of steps on a few points, as in test_icnn.py.
"""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from maesolvers import (  # noqa: E402
    ICNN,
    LBFGS,
    T1,
    T12,
    T16,
    Adam,
    Disk,
    HausdorffLoss,
    IdentityStart,
    Problem,
    Switch,
    Transport,
    TransportICNN,
    TransportSampler,
    UnsupportedProblemError,
)
from maesolvers.core import Ellipse  # noqa: E402
from maesolvers.metrics import gradient_fd, gradient_map_error  # noqa: E402
from maesolvers.neural import Batch, NeuralSolver  # noqa: E402
from maesolvers.neural import functional as F  # noqa: E402
from maesolvers.neural._torch import building  # noqa: E402

pytestmark = pytest.mark.neural

SMALL = {
    "interior_points": 40,
    "boundary_points": 30,
    "target_points": 30,
    "fit_steps": 30,
    "accelerator": "cpu",
}


def _tensor(array):
    return torch.tensor(np.asarray(array, dtype=float), dtype=torch.float64)


class Exact(torch.nn.Module):
    """T12's solution ``x1^2 + 3.5 x1 + x2^2 / 4``, scaled by ``scale``."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))

    def forward(self, x):
        return self.scale * (x[:, 0] ** 2 + 3.5 * x[:, 0] + 0.25 * x[:, 1] ** 2)


def _batch(boundary_angles, target_angles, interior=None, target=True):
    """A T12 batch: points on the unit circle and on the ellipse, by angle."""
    t = np.asarray(boundary_angles, dtype=float)
    s = np.asarray(target_angles, dtype=float)
    circle = np.stack([np.cos(t), np.sin(t)], axis=-1)
    ellipse = np.stack([3.5 + 2 * np.cos(s), 0.5 * np.sin(s)], axis=-1)
    inside = np.array([[0.1, 0.2], [-0.3, 0.4]]) if interior is None else interior
    return Batch(
        interior=_tensor(inside),
        f=_tensor(T12.f(inside)),
        boundary=_tensor(circle),
        g=None,
        f_boundary=_tensor(T12.f(circle)),
        target=T12.boundary.target,
        target_boundary=_tensor(ellipse) if target else None,
    )


# The kernel


def test_nearest_squared_distances_match_brute_force():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(7, 2)), rng.normal(size=(5, 2))
    into, onto = F.nearest_squared_distances(_tensor(a), _tensor(b))
    squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    np.testing.assert_allclose(into.numpy(), squared.min(axis=1), rtol=1e-14)
    np.testing.assert_allclose(onto.numpy(), squared.min(axis=0), rtol=1e-14)


def test_nearest_squared_distances_have_a_finite_gradient_at_coincident_points():
    a = _tensor([[1.0, 2.0], [0.0, 0.0]]).requires_grad_(True)
    b = _tensor([[1.0, 2.0]])
    into, onto = F.nearest_squared_distances(a, b)
    (into.sum() + onto.sum()).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    np.testing.assert_array_equal(a.grad[0].numpy(), [0.0, 0.0])


def test_nearest_squared_distances_refuse_an_empty_set():
    with pytest.raises(ValueError, match="two non-empty sets, got 0 and 1"):
        F.nearest_squared_distances(torch.zeros(0, 2), torch.zeros(1, 2))


# The loss


def test_every_term_vanishes_at_the_exact_solution():
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    terms = HausdorffLoss()(Exact(), _batch(angles, angles))
    assert set(terms) == {"loss", "equation", "into", "onto"}
    for value in terms.values():
        assert float(value) == pytest.approx(0.0, abs=1e-26)


def test_the_terms_are_means_of_squared_distances_and_weighted():
    # u = 2 x scaled: grad u(x) = 2 grad u_exact; det D^2 u = 4.
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    batch = _batch(angles, angles)
    u = Exact(scale=2.0)
    terms = HausdorffLoss(boundary_weight=3.0)(u, batch)
    assert float(terms["equation"]) == pytest.approx(9.0)  # (4 - 1)^2
    image = 2 * np.stack([2 * np.cos(angles) + 3.5, 0.5 * np.sin(angles)], axis=-1)
    target = batch.target_boundary.numpy()
    squared = np.sum((image[:, None] - target[None]) ** 2, axis=-1)
    assert float(terms["into"]) == pytest.approx(squared.min(axis=1).mean())
    assert float(terms["onto"]) == pytest.approx(squared.min(axis=0).mean())
    total = terms["equation"] + 3.0 * (terms["into"] + terms["onto"])
    assert float(terms["loss"]) == pytest.approx(float(total))
    terms["loss"].backward()
    assert u.scale.grad is not None and float(u.scale.grad) > 0


def test_the_onto_term_sees_a_map_that_misses_part_of_the_target():
    # The boundary points cover half the circle: their images lie on the
    # ellipse (into = 0) but leave its other half uncovered (onto > 0).
    half = np.linspace(0, np.pi, 9)
    terms = HausdorffLoss()(Exact(), _batch(half, np.linspace(0, 2 * np.pi, 17)[:-1]))
    assert float(terms["into"]) == pytest.approx(0.0, abs=1e-26)
    assert float(terms["onto"]) > 0.1


def test_the_loss_needs_boundary_and_target_points():
    angles = np.linspace(0, 2 * np.pi, 5)
    with pytest.raises(ValueError, match="target boundary points; use a sampler"):
        HausdorffLoss()(Exact(), _batch(angles, angles, target=False))
    with pytest.raises(ValueError, match="needs boundary points"):
        HausdorffLoss()(Exact(), _batch([], angles))


def test_the_loss_supports_transport_problems_and_checks_its_weight():
    assert HausdorffLoss().supports(T12) and HausdorffLoss().supports(T16)
    assert not HausdorffLoss().supports(T1)
    with pytest.raises(ValueError, match="boundary_weight must be positive"):
        HausdorffLoss(boundary_weight=0.0)


# The start


def test_the_identity_start_fits_the_gradient_to_the_identity_map():
    collocation = TransportSampler(
        interior_points=50, boundary_points=20, target_points=1
    ).sample(T12, np.random.default_rng(0))
    part = IdentityStart(network=ICNN(width=6, depth=2), fit_steps=500, fit_tol=1e-8)
    with building(0):
        module = part.build(T12, collocation)
    report = module.report["identity_fit"]
    assert set(report) == {
        "iterations",
        "max_gradient",
        "converged",
        "mse",
        "max_misfit",
    }
    points = np.concatenate([collocation.interior, collocation.boundary])
    misfit = F.gradient(module, _tensor(points), create_graph=False) - _tensor(points)
    assert float(misfit.abs().max()) == pytest.approx(report["max_misfit"])
    assert report["max_misfit"] < 0.05
    assert all(p.requires_grad for p in module.parameters())


def test_the_identity_start_delegates_support_and_checks_its_fields():
    assert IdentityStart(network=ICNN()).supports(T12)
    with pytest.raises(TypeError, match=r"IdentityStart\.network"):
        IdentityStart(network=ICNN)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fit_steps must be at least 1"):
        IdentityStart(network=ICNN(), fit_steps=0)
    with pytest.raises(ValueError, match="fit_tol must be positive"):
        IdentityStart(network=ICNN(), fit_tol=0.0)


# The optimizer


def test_switch_gives_the_first_steps_to_the_first_optimizer():
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0], dtype=torch.float64))
    optimizer = Switch(first=Adam(lr=0.1), then=LBFGS(), after=2).build([x])

    def closure():
        optimizer.zero_grad()
        loss = (x**2).sum()
        loss.backward()
        return loss

    for _ in range(4):
        optimizer.step(closure)
    assert optimizer.first.state[x]["step"] == 2
    assert optimizer.then.state[x]["n_iter"] == 2
    assert float((x**2).sum()) < 1e-10  # L-BFGS solves the quadratic


def test_switch_checks_its_parts_and_its_count():
    with pytest.raises(TypeError, match=r"Switch\.then"):
        Switch(first=Adam(), then=LBFGS, after=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after must be at least 0"):
        Switch(first=Adam(), then=LBFGS(), after=-1)
    assert Switch(first=Adam(), then=LBFGS(), after=0).after == 0


# The preset


def test_the_preset_is_the_paper_composition():
    preset = TransportICNN()
    assert preset.network == IdentityStart(
        network=ICNN(width=10, depth=4), fit_steps=1000, fit_tol=1e-6
    )
    assert preset.loss == HausdorffLoss(boundary_weight=1.0)
    assert preset.sampler == TransportSampler(
        interior_points=800, boundary_points=800, target_points=800
    )
    assert preset.optimizer == Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200)
    assert (preset.max_steps, preset.tol) == (2000, 1e-7)


def test_the_preset_records_flat_hyperparameters():
    init_args = TransportICNN(adam_steps=2000, seed=4).hparams["init_args"]
    assert init_args["adam_steps"] == 2000 and init_args["seed"] == 4
    assert set(init_args) == {
        "max_steps",
        "tol",
        "seed",
        "accelerator",
        "width",
        "depth",
        "init_std",
        "fit_steps",
        "fit_tol",
        "boundary_weight",
        "interior_points",
        "boundary_points",
        "target_points",
        "adam_lr",
        "adam_steps",
    }


def test_the_preset_supports_transport_problems_only():
    preset = TransportICNN()
    assert preset.supports(T12) and preset.supports(T16)
    assert not preset.supports(T1)
    with pytest.raises(UnsupportedProblemError):
        preset.solve(T1)


def test_the_preset_refuses_a_problem_out_of_mass_balance():
    heavy = dataclasses.replace(T12, name="heavy", f=lambda x: 2.0 + 0.0 * x[..., 0])
    with pytest.raises(ValueError, match="mass"):
        TransportICNN(**SMALL).solve(heavy)


def test_a_short_training_moves_the_map_towards_the_exact_one():
    preset = TransportICNN(max_steps=6, adam_steps=3, **SMALL)
    result = preset.solve(T12)
    info = result.info
    assert info["stop"] == "max_steps" and info["iterations"] == 6
    assert set(info["loss_terms"]) == {"loss", "equation", "into", "onto"}
    assert info["collocation"]["target_boundary_points"] == 30
    assert "identity_fit" in info["network"]
    points = preset.collocation(T12).interior
    error = gradient_map_error(
        gradient_fd(result.u, points), gradient_fd(T12.exact, points)
    )
    # The identity map is about 3.5 from the exact one; six steps get well below.
    assert error["gradient_rms"] < 1.5
    assert info["history"][-1] < info["history"][0]


def test_the_preset_takes_an_ellipse_as_source_and_target():
    ellipse = Ellipse((0.0, 0.0), (2.0, 0.5))
    problem = Problem(
        "E", ellipse, lambda x: 1.0 + 0.0 * x[..., 0], Transport(Disk((0, 0), 1))
    )
    assert TransportICNN().supports(problem)
    result = TransportICNN(max_steps=2, adam_steps=1, **SMALL).solve(problem)
    assert result.info["iterations"] == 2


def test_the_parts_compose_with_another_network_and_optimizer():
    solver = NeuralSolver(
        network=ICNN(width=4, depth=2),
        loss=HausdorffLoss(),
        sampler=TransportSampler(
            interior_points=20, boundary_points=10, target_points=10
        ),
        optimizer=Adam(lr=1e-2),
        max_steps=3,
    )
    assert solver.supports(T12) and not solver.supports(T1)
    assert solver.solve(T12).info["iterations"] == 3
