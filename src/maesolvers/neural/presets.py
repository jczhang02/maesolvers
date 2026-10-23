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

from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Dirichlet, Transport
from maesolvers.core.domains import Box, CutQuarterDisk, Disk, Ellipse
from maesolvers.core.problem import Problem
from maesolvers.core.transport import check_mass_balance
from maesolvers.neural.losses import HausdorffLoss, Loss, PenaltyLoss
from maesolvers.neural.networks import ICNN, MLP, HardDirichlet, IdentityStart, Network
from maesolvers.neural.optimizers import BFGS, LBFGS, Adam, Optimizer, Switch
from maesolvers.neural.samplers import Sampler, TransportSampler, UniformSampler
from maesolvers.neural.solver import NeuralSolver


@dataclass(frozen=True, kw_only=True)
class PenaltyMLP(NeuralSolver):
    """The penalty MLP of Nystrom-Vestberg 2023 (Section 3.1, Example 4.1).

    ``PenaltyMLP(width=w, depth=k, convexity_weight=C, interior_points=M,
    boundary_points=K, fit_steps=s, fit_tol=t, ...)`` is ``NeuralSolver``
    with

    - ``network=HardDirichlet(network=MLP(width=w, depth=k,
      activation="sigmoid"), fit_steps=s, fit_tol=t)``: the ansatz
      ``u = G + D N``;
    - ``loss=PenaltyLoss(convexity_weight=C)``: the squared residual summed
      over the interior points plus ``C`` times the mean diagonal penalty;
    - ``sampler=UniformSampler(interior_points=M, boundary_points=K)``;
    - ``optimizer=LBFGS()``.

    The defaults are the parts' defaults, which are the paper's except for
    what the paper does not give: the cap ``fit_steps`` of the fits of
    ``G`` and ``D``, the step budget ``max_steps`` (the 5000 of the paper's
    Example 4.6) and the L-BFGS memory (``docs/methods/penalty_mlp.md``,
    Differences from the paper). The training runs on the CPU unless
    ``accelerator`` says otherwise, so the seed fixes the result on a given
    machine and thread count. The penalty makes the solution convex only in
    2D, so the preset is 2D only; it needs the domain's boundary sampling
    and distance (:class:`~maesolvers.core.domains.BoundarySampling`,
    :class:`~maesolvers.core.domains.BoundaryDistance`).

    Parameters
    ----------
    width, depth
        Units per hidden layer and hidden layers of ``N``, each at least 1
        (10 and 5: 481 parameters).
    convexity_weight
        ``C``, at least 0.
    interior_points, boundary_points
        Collocation points inside the domain (at least 1) and on its
        boundary (at least 1: the ansatz is fitted there).
    fit_steps, fit_tol
        Most L-BFGS iterations of each fit of ``G`` and ``D``, and the
        largest gradient component at which a fit stops.
    max_steps
        Most L-BFGS iterations.
    tol
        Stop when the largest component of the loss gradient is at most this.
    seed
        Seeds the collocation points and the initial networks.
    accelerator
        ``"cpu"``, ``"gpu"`` or ``"auto"`` (a GPU when there is one).
    """

    domains: ClassVar[tuple[type, ...]] = (Box, Disk, CutQuarterDisk)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...] | None] = (2,)

    network: Network = field(init=False, repr=False)
    loss: Loss = field(init=False, repr=False)
    sampler: Sampler = field(init=False, repr=False)
    optimizer: Optimizer = field(init=False, repr=False)
    width: int = MLP.width
    depth: int = MLP.depth
    convexity_weight: float = PenaltyLoss.convexity_weight
    interior_points: int = UniformSampler.interior_points
    boundary_points: int = UniformSampler.boundary_points
    fit_steps: int = HardDirichlet.fit_steps
    fit_tol: float = HardDirichlet.fit_tol
    max_steps: int = 5000

    def __post_init__(self) -> None:
        boundary_points = integer("boundary_points", self.boundary_points, 1)
        mlp = MLP(width=self.width, depth=self.depth, activation="sigmoid")
        network = HardDirichlet(
            network=mlp, fit_steps=self.fit_steps, fit_tol=self.fit_tol
        )
        loss = PenaltyLoss(convexity_weight=self.convexity_weight)
        sampler = UniformSampler(
            interior_points=self.interior_points, boundary_points=boundary_points
        )
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "optimizer", LBFGS())
        object.__setattr__(self, "width", mlp.width)
        object.__setattr__(self, "depth", mlp.depth)
        object.__setattr__(self, "convexity_weight", loss.convexity_weight)
        object.__setattr__(self, "interior_points", sampler.interior_points)
        object.__setattr__(self, "boundary_points", sampler.boundary_points)
        object.__setattr__(self, "fit_steps", network.fit_steps)
        object.__setattr__(self, "fit_tol", network.fit_tol)
        super().__post_init__()


@dataclass(frozen=True, kw_only=True)
class DirichletICNN(NeuralSolver):
    """The input convex network of Nystrom-Vestberg 2023 (Section 3.2, Example 4.2).

    ``DirichletICNN(width=w, depth=k, boundary_weight=C, interior_points=M,
    boundary_points=K, ...)`` is ``NeuralSolver`` with

    - ``network=ICNN(width=w, depth=k)``: convex by construction, so the
      loss needs no convexity term;
    - ``loss=PenaltyLoss(convexity_weight=0, boundary_weight=C,
      equation_at_boundary=True)``: the squared residual summed over the
      interior and the boundary points plus ``C`` times the mean squared
      boundary misfit, eq. (3.8) as Example 4.2 uses it;
    - ``sampler=UniformSampler(interior_points=M, boundary_points=K)``;
    - ``optimizer=BFGS()``: dense BFGS, as the paper's SciPy BFGS.

    The defaults are the paper's (5 x 10 softplus, 563 parameters in 2D,
    ``C = 100``, 3000 and 400 points), except for what it does not give: the
    step budget ``max_steps`` is the "at most 10,000 iterations" of its
    Example 4.6, and ``init_std`` is 1 (``docs/methods/icnn.md``,
    Differences from the paper). Nothing in the
    composition depends on the dimension, so the preset takes a Dirichlet
    problem in any dimension on a domain that samples its boundary
    (:class:`~maesolvers.core.domains.BoundarySampling`). The training runs
    on the CPU unless ``accelerator`` says otherwise.

    Parameters
    ----------
    width, depth
        Units per hidden layer and hidden layers of the network, each at
        least 1.
    init_std
        Standard deviation of the initial parameters of the network,
        positive.
    boundary_weight
        ``C``, the weight of the boundary misfit, positive: with no ansatz
        to meet the boundary condition, this term alone imposes it.
    interior_points, boundary_points
        Collocation points inside the domain (at least 1) and on its
        boundary (at least 1: the boundary condition is imposed there).
    max_steps
        Most training steps, one BFGS iteration each.
    tol
        Stop when the largest component of the loss gradient is at most this.
    seed
        Seeds the collocation points and the initial network.
    accelerator
        ``"cpu"``, ``"gpu"`` or ``"auto"`` (a GPU when there is one).
    """

    domains: ClassVar[tuple[type, ...]] = (Box, Disk, Ellipse, CutQuarterDisk)
    boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)
    dims: ClassVar[tuple[int, ...] | None] = None

    network: Network = field(init=False, repr=False)
    loss: Loss = field(init=False, repr=False)
    sampler: Sampler = field(init=False, repr=False)
    optimizer: Optimizer = field(init=False, repr=False)
    width: int = ICNN.width
    depth: int = ICNN.depth
    init_std: float = ICNN.init_std
    boundary_weight: float = 100.0
    interior_points: int = UniformSampler.interior_points
    boundary_points: int = UniformSampler.boundary_points
    max_steps: int = 10_000

    def __post_init__(self) -> None:
        boundary_points = integer("boundary_points", self.boundary_points, 1)
        network = ICNN(width=self.width, depth=self.depth, init_std=self.init_std)
        loss = PenaltyLoss(
            convexity_weight=0.0,
            boundary_weight=positive("boundary_weight", self.boundary_weight),
            equation_at_boundary=True,
        )
        sampler = UniformSampler(
            interior_points=self.interior_points, boundary_points=boundary_points
        )
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "optimizer", BFGS())
        object.__setattr__(self, "width", network.width)
        object.__setattr__(self, "depth", network.depth)
        object.__setattr__(self, "init_std", network.init_std)
        object.__setattr__(self, "boundary_weight", loss.boundary_weight)
        object.__setattr__(self, "interior_points", sampler.interior_points)
        object.__setattr__(self, "boundary_points", sampler.boundary_points)
        super().__post_init__()


@dataclass(frozen=True, kw_only=True)
class TransportICNN(NeuralSolver):
    """The convex PINN of Caboussat-Peruso 2025: an input convex network on a transport problem.

    ``TransportICNN(width=w, depth=k, init_std=s, fit_steps=n, fit_tol=t,
    boundary_weight=C, interior_points=M, boundary_points=K,
    target_points=L, adam_lr=r, adam_steps=a, ...)`` is ``NeuralSolver``
    with

    - ``network=IdentityStart(network=ICNN(width=w, depth=k, init_std=s),
      fit_steps=n, fit_tol=t)``: convex by construction, and fitted to
      ``grad u = x`` before the training (Section 4.1);
    - ``loss=HausdorffLoss(boundary_weight=C)``: the mean squared residual
      plus ``C`` times the Hausdorff-type distance between ``grad u`` of the
      boundary points and the target's boundary points, eq. (3.4);
    - ``sampler=TransportSampler(interior_points=M, boundary_points=K,
      target_points=L)``;
    - ``optimizer=Switch(first=Adam(lr=r), then=LBFGS(), after=a)``: Adam
      for ``a`` steps, then L-BFGS with a strong Wolfe line search and step
      1 (Table 1), one iteration per step.

    The defaults are the paper's where it gives them: 4 hidden layers of 10
    softplus units (433 parameters in 2D), ``C = 1``, 800 interior and 800
    boundary points, the switch to L-BFGS after 200 of 2000 steps (Figure
    5; Table 1's 100 epochs of 20 L-BFGS iterations). What it does not
    give is ``docs/methods/icnn.md``'s list: the Adam learning rate, the
    initial parameters, the fit to the identity map, the number of target
    points and the stopping test. ``adam_steps`` at least ``max_steps``
    gives the paper's Adam-only runs (Table 2, "Adam").

    The preset takes a :class:`~maesolvers.core.boundary.Transport`
    problem in any dimension whose domain and target are each a ``Box``,
    a ``Disk`` or an ``Ellipse``: the sets whose boundary the sampler
    draws and whose source mass and volume
    :func:`~maesolvers.core.transport.check_mass_balance` computes. Its
    :meth:`supports` is False on another domain or target, such as a
    ``CutQuarterDisk``, on which the composed ``NeuralSolver`` with the
    same parts still runs, with the caller checking the mass. Its
    :meth:`check` raises ``ValueError`` when the source mass differs from
    the target's volume. The training runs on the CPU unless
    ``accelerator`` says otherwise.

    Parameters
    ----------
    width, depth
        Units per hidden layer and hidden layers of the network, each at
        least 1.
    init_std
        Standard deviation of the network's parameters before the fit,
        positive.
    fit_steps, fit_tol
        Most L-BFGS iterations of the fit to the identity map, and the
        largest gradient component at which it stops.
    boundary_weight
        ``C``, the weight of the transport term, positive.
    interior_points, boundary_points, target_points
        Collocation points inside the domain, on its boundary and on the
        target's boundary, each at least 1.
    adam_lr
        Learning rate of Adam, positive.
    adam_steps
        Steps of Adam before L-BFGS, at least 0.
    max_steps
        Most training steps, Adam's and L-BFGS's together.
    tol
        Stop when the largest component of the loss gradient is at most this.
    seed
        Seeds the collocation points and the initial network.
    accelerator
        ``"cpu"``, ``"gpu"`` or ``"auto"`` (a GPU when there is one).
    """

    domains: ClassVar[tuple[type, ...]] = (Box, Disk, Ellipse)
    boundaries: ClassVar[tuple[type, ...]] = (Transport,)
    dims: ClassVar[tuple[int, ...] | None] = None

    network: Network = field(init=False, repr=False)
    loss: Loss = field(init=False, repr=False)
    sampler: Sampler = field(init=False, repr=False)
    optimizer: Optimizer = field(init=False, repr=False)
    width: int = ICNN.width
    depth: int = 4
    init_std: float = ICNN.init_std
    fit_steps: int = IdentityStart.fit_steps
    fit_tol: float = IdentityStart.fit_tol
    boundary_weight: float = HausdorffLoss.boundary_weight
    interior_points: int = TransportSampler.interior_points
    boundary_points: int = TransportSampler.boundary_points
    target_points: int = TransportSampler.target_points
    adam_lr: float = Adam.lr
    adam_steps: int = 200
    max_steps: int = 2000
    tol: float = 1e-7

    def __post_init__(self) -> None:
        icnn = ICNN(width=self.width, depth=self.depth, init_std=self.init_std)
        network = IdentityStart(
            network=icnn, fit_steps=self.fit_steps, fit_tol=self.fit_tol
        )
        loss = HausdorffLoss(boundary_weight=self.boundary_weight)
        sampler = TransportSampler(
            interior_points=self.interior_points,
            boundary_points=self.boundary_points,
            target_points=self.target_points,
        )
        adam = Adam(lr=self.adam_lr)
        optimizer = Switch(first=adam, then=LBFGS(), after=self.adam_steps)
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "width", icnn.width)
        object.__setattr__(self, "depth", icnn.depth)
        object.__setattr__(self, "init_std", icnn.init_std)
        object.__setattr__(self, "fit_steps", network.fit_steps)
        object.__setattr__(self, "fit_tol", network.fit_tol)
        object.__setattr__(self, "boundary_weight", loss.boundary_weight)
        object.__setattr__(self, "interior_points", sampler.interior_points)
        object.__setattr__(self, "boundary_points", sampler.boundary_points)
        object.__setattr__(self, "target_points", sampler.target_points)
        object.__setattr__(self, "adam_lr", adam.lr)
        object.__setattr__(self, "adam_steps", optimizer.after)
        super().__post_init__()

    def supports(self, problem: Problem) -> bool:
        """Whether the parts support the problem and its domain and target are in ``domains``.

        The class's ``domains`` are the sets whose mass :meth:`check` can
        compute; the parts alone would also take a ``CutQuarterDisk``.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        boundary = problem.boundary
        return (
            super().supports(problem)
            and isinstance(boundary, Transport)
            and isinstance(problem.domain, self.domains)
            and isinstance(boundary.target, self.domains)
        )

    def check(self, problem: Problem) -> None:
        """Raise if the parts do not support the problem or its mass is out of balance.

        Parameters
        ----------
        problem
            The problem to solve.

        Raises
        ------
        UnsupportedProblemError
            If :meth:`supports` is False.
        ValueError
            If the integral of ``f`` over the domain differs from the
            target's volume by more than a relative ``1e-3``
            (:func:`~maesolvers.core.transport.check_mass_balance`).
        ImportError
            Without torch or lightning.
        """
        super().check(problem)
        check_mass_balance(problem)
