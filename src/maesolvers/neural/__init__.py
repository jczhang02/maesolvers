"""Neural solvers: a network, a loss, a sampler and an optimizer on one training loop.

:class:`NeuralSolver` composes the four parts (ADR 0004) and trains them on
the shared loop of ADR 0008, a Lightning ``Trainer`` in float64;
:class:`PenaltyMLP` is the preset for the penalty MLP of Nystrom-Vestberg
2023, and :class:`DirichletICNN` for its input convex network
(:class:`ICNN`). A new part subclasses :class:`Network`, :class:`Loss`,
:class:`Sampler` or :class:`Optimizer` as a frozen dataclass;
:mod:`maesolvers.neural.functional` holds the torch kernels a loss is built
from.

Importing this package needs numpy only, so the solvers can be configured
and listed without the ``neural`` extra; torch and lightning are imported
when a solve starts, and ``solve`` raises an ImportError that names the
extra without them.
"""

from maesolvers.neural.losses import Batch, Loss, PenaltyLoss
from maesolvers.neural.networks import ICNN, MLP, HardDirichlet, Network
from maesolvers.neural.optimizers import LBFGS, Adam, Optimizer
from maesolvers.neural.presets import DirichletICNN, PenaltyMLP
from maesolvers.neural.samplers import (
    Collocation,
    Sampler,
    UniformSampler,
    sample_interior,
)
from maesolvers.neural.solver import NeuralSolver

__all__ = [
    "ICNN",
    "LBFGS",
    "MLP",
    "Adam",
    "Batch",
    "Collocation",
    "DirichletICNN",
    "HardDirichlet",
    "Loss",
    "Network",
    "NeuralSolver",
    "Optimizer",
    "PenaltyLoss",
    "PenaltyMLP",
    "Sampler",
    "UniformSampler",
    "sample_interior",
]
