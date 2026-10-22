"""Presets: named neural solvers for the parts of a source paper.

A preset (ADR 0004) is a :class:`~maesolvers.neural.solver.NeuralSolver`
whose parts are built from flat parameters, so the simple call
``PenaltyMLP().solve(problem)`` stays and its
:attr:`~maesolvers.core.solver.Solver.hparams` stay flat. The presets are
what :data:`maesolvers.registry.SOLVERS` and the comparison tables list, and
like every preset they declare their support on the class.
"""

# No `from __future__ import annotations` here: jsonargparse reads the field
# types (ADR 0005); see maesolvers.classical.iterations.

from dataclasses import dataclass, field
from typing import ClassVar

from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import Box, CutQuarterDisk, Disk
from maesolvers.neural.losses import Loss, PenaltyLoss
from maesolvers.neural.networks import MLP, HardDirichlet, Network
from maesolvers.neural.optimizers import LBFGS, Optimizer
from maesolvers.neural.samplers import Sampler, UniformSampler
from maesolvers.neural.solver import NeuralSolver


@dataclass(frozen=True, kw_only=True)
class PenaltyMLP(NeuralSolver):
    """The penalty MLP of Nystrom-Vestberg 2023 (Section 3.1, Example 4.1).

    ``PenaltyMLP(width=w, depth=k, convexity_weight=C, interior=M,
    boundary=K, ...)`` is ``NeuralSolver`` with

    - ``network=HardDirichlet(network=MLP(width=w, depth=k,
      activation="sigmoid"))``: the ansatz ``u = G + D N``;
    - ``loss=PenaltyLoss(convexity_weight=C)``: the squared residual summed
      over the interior points plus ``C`` times the mean diagonal penalty;
    - ``sampler=UniformSampler(interior=M, boundary=K)``;
    - ``optimizer=LBFGS()``.

    The defaults are the paper's. The penalty makes the solution convex only
    in 2D, so the preset is 2D only; it needs the domain's boundary sampling
    and distance (:class:`~maesolvers.core.domains.BoundarySampling`,
    :class:`~maesolvers.core.domains.BoundaryDistance`).

    Parameters
    ----------
    width, depth
        Units per hidden layer and hidden layers of ``N``, each at least 1
        (10 and 5: 481 parameters).
    convexity_weight
        ``C``, at least 0.
    interior, boundary
        Collocation points inside the domain (at least 1) and on its boundary
        (at least 1: the ansatz is fitted there).
    max_steps
        Most L-BFGS iterations.
    tol
        Stop when the largest component of the loss gradient is at most this.
    seed
        Seeds the collocation points and the initial networks.
    accelerator
        ``"auto"``, ``"cpu"`` or ``"gpu"``.
    """

    domains: ClassVar[tuple[type, ...]] = (Box, Disk, CutQuarterDisk)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...] | None] = (2,)

    network: Network = field(init=False, repr=False)
    loss: Loss = field(init=False, repr=False)
    sampler: Sampler = field(init=False, repr=False)
    optimizer: Optimizer = field(init=False, repr=False)
    width: int = 10
    depth: int = 5
    convexity_weight: float = 1e4
    interior: int = 3000
    boundary: int = 400
    max_steps: int = 5000

    def __post_init__(self) -> None:
        mlp = MLP(width=self.width, depth=self.depth, activation="sigmoid")
        loss = PenaltyLoss(convexity_weight=self.convexity_weight)
        sampler = UniformSampler(interior=self.interior, boundary=self.boundary)
        if sampler.boundary == 0:
            raise ValueError("boundary must be at least 1, got 0")
        object.__setattr__(self, "network", HardDirichlet(network=mlp))
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "optimizer", LBFGS())
        object.__setattr__(self, "width", mlp.width)
        object.__setattr__(self, "depth", mlp.depth)
        object.__setattr__(self, "convexity_weight", loss.convexity_weight)
        object.__setattr__(self, "interior", sampler.interior)
        object.__setattr__(self, "boundary", sampler.boundary)
        super().__post_init__()
