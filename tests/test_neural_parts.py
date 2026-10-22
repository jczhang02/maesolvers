"""The neural solver and its parts as configuration: numpy only, no training."""

import dataclasses
import sys

import numpy as np
import pytest

from maesolvers import (
    CATALOGUE,
    LBFGS,
    MLP,
    SOLVERS,
    T1,
    T6,
    T8,
    T9,
    T12,
    Adam,
    Dirichlet,
    HardDirichlet,
    NeuralSolver,
    PenaltyLoss,
    PenaltyMLP,
    Problem,
    UniformSampler,
    UnsupportedProblemError,
)
from maesolvers.core import Ellipse
from maesolvers.neural import Loss, Network, Sampler


def _solver(**kwargs):
    parts = {
        "network": HardDirichlet(network=MLP()),
        "loss": PenaltyLoss(),
        "sampler": UniformSampler(),
        "optimizer": LBFGS(),
    }
    return NeuralSolver(**{**parts, **kwargs})


def test_the_preset_is_the_paper_composition():
    preset = PenaltyMLP()
    assert preset.network == HardDirichlet(
        network=MLP(width=10, depth=5, activation="sigmoid"),
        lift_width=20,
        distance_width=40,
        distance_points=300,
    )
    assert preset.loss == PenaltyLoss(convexity_weight=1e4, boundary_weight=0.0)
    assert preset.sampler == UniformSampler(interior=3000, boundary=400)
    assert preset.optimizer == LBFGS()
    composed = _solver(max_steps=5000)
    assert all(
        getattr(preset, f.name) == getattr(composed, f.name)
        for f in dataclasses.fields(NeuralSolver)
    )


def test_the_preset_records_flat_hyperparameters():
    assert PenaltyMLP(width=8, seed=3).hparams == {
        "class_path": "maesolvers.PenaltyMLP",
        "init_args": {
            "max_steps": 5000,
            "tol": 1e-5,
            "seed": 3,
            "accelerator": "auto",
            "width": 8,
            "depth": 5,
            "convexity_weight": 1e4,
            "interior": 3000,
            "boundary": 400,
        },
    }


def test_the_composed_solver_records_its_parts():
    config = _solver(optimizer=Adam(lr=0.01)).hparams
    assert config["class_path"] == "maesolvers.NeuralSolver"
    assert config["init_args"]["optimizer"] == {
        "class_path": "maesolvers.Adam",
        "init_args": {"lr": 0.01},
    }
    assert config["init_args"]["network"]["init_args"]["network"] == {
        "class_path": "maesolvers.MLP",
        "init_args": {"width": 10, "depth": 5, "activation": "sigmoid"},
    }


def test_the_registry_lists_the_penalty_mlp():
    assert SOLVERS["PenaltyMLP"] is PenaltyMLP


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_the_preset_class_declaration_agrees_with_its_parts(name):
    problem = CATALOGUE[name]
    declared = (
        isinstance(problem.domain, PenaltyMLP.domains)
        and isinstance(problem.boundary, PenaltyMLP.boundaries)
        and problem.dim in PenaltyMLP.dims
    )
    assert PenaltyMLP().supports(problem) == declared


def test_the_penalty_mlp_supports_2d_dirichlet_problems_on_sampled_domains():
    assert PenaltyMLP().supports(T1) and PenaltyMLP().supports(T6)
    assert PenaltyMLP().supports(T8)
    assert not PenaltyMLP().supports(T9)  # 3D: the penalty does not imply convexity
    assert not PenaltyMLP().supports(T12)  # transport
    ellipse = Problem("E", Ellipse((0, 0), (2, 1)), T1.f, Dirichlet(T1.f))
    assert not PenaltyMLP().supports(ellipse)  # no boundary sampling or distance


def test_the_composed_solver_supports_what_all_its_parts_support():
    plain = _solver(network=MLP(), loss=PenaltyLoss(convexity_weight=0.0))
    assert plain.supports(T9)  # a plain MLP and no penalty: any dimension
    assert not _solver(loss=PenaltyLoss(convexity_weight=0.0)).supports(T12)
    with pytest.raises(
        UnsupportedProblemError, match="NeuralSolver does not support T9"
    ):
        _solver().solve(T9)


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (lambda: MLP(width=0), ValueError, "width must be at least 1"),
        (lambda: MLP(depth=2.0), TypeError, "depth must be an integer"),
        (lambda: MLP(activation="relu"), ValueError, "activation must be"),
        (lambda: HardDirichlet(network=MLP, fit_steps=1), TypeError, "write MLP"),
        (lambda: HardDirichlet(network=PenaltyLoss()), TypeError, "subclass Network"),
        (lambda: HardDirichlet(network=MLP(), fit_steps=0), ValueError, "fit_steps"),
        (lambda: PenaltyLoss(convexity_weight=-1), ValueError, "nonnegative"),
        (lambda: PenaltyLoss(boundary_weight="1"), TypeError, "real number"),
        (lambda: LBFGS(lr=0), ValueError, "lr must be positive"),
        (lambda: LBFGS(history_size=0), ValueError, "history_size"),
        (lambda: Adam(lr=float("nan")), ValueError, "lr must be positive"),
        (lambda: _solver(max_steps=0), ValueError, "max_steps must be at least 1"),
        (lambda: _solver(tol=0.0), ValueError, "tol must be positive"),
        (lambda: _solver(seed=-1), ValueError, "seed must be at least 0"),
        (lambda: _solver(accelerator="tpu"), ValueError, "accelerator must be"),
        (lambda: _solver(sampler=MLP()), TypeError, "sampler must subclass Sampler"),
        (lambda: PenaltyMLP(boundary=0), ValueError, "boundary must be at least 1"),
        (lambda: PenaltyMLP(width=1.5), TypeError, "width must be an integer"),
    ],
)
def test_parts_and_solvers_check_their_parameters(make, error, message):
    with pytest.raises(error, match=message):
        make()


def test_a_part_needs_its_own_frozen_dataclass_decorator():
    class Loose(Loss):  # no decorator
        def __call__(self, u, batch):
            return {}

    with pytest.raises(
        TypeError, match=r"NeuralSolver\.loss: Loose must carry its own"
    ):
        _solver(loss=Loose())


def test_parts_store_plain_numbers():
    mlp = MLP(width=np.int64(4))
    assert type(mlp.width) is int
    assert type(PenaltyLoss(convexity_weight=np.float32(2)).convexity_weight) is float
    assert type(PenaltyMLP(interior=np.int32(10)).interior) is int


def test_the_collocation_points_follow_the_seed_and_the_sampler():
    solver = PenaltyMLP(interior=40, boundary=12)
    first = solver.collocation(T6)
    np.testing.assert_array_equal(first.interior, solver.collocation(T6).interior)
    other = PenaltyMLP(interior=40, boundary=12, seed=1).collocation(T6)
    assert first.fingerprint() != other.fingerprint()
    # Another method with the same sampler and seed trains on the same points.
    plain = _solver(network=MLP(), sampler=UniformSampler(interior=40, boundary=12))
    assert plain.collocation(T6).fingerprint() == first.fingerprint()


def test_solving_without_the_neural_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import fails
    with pytest.raises(ImportError, match=r'"maesolvers\[neural\] @ git\+'):
        PenaltyMLP().solve(T1)


def test_the_part_bases_are_abstract():
    for base in (Network, Loss, Sampler):
        with pytest.raises(TypeError):
            base()
