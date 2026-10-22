"""Networks: the part of a neural solver that represents the solution.

A network part (ADR 0004) is a frozen dataclass of hyperparameters whose
:meth:`Network.build` makes the torch module ``u`` for one problem. The
module maps points of shape ``(M, d)`` to values of shape ``(M,)``. The
solver calls ``build`` once per solve, with float64 as torch's default
dtype and torch's random generator seeded from the solver's seed, on the
CPU; the training loop moves the module to its device.

This module imports torch only inside ``build``, so the parts can be
configured, compared and recorded without the ``neural`` extra.
"""

# No `from __future__ import annotations` in this module: jsonargparse reads the
# field types of the parts (ADR 0005); see maesolvers.classical.iterations.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from maesolvers.core._params import integer
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import BoundaryDistance
from maesolvers.core.problem import Problem
from maesolvers.core.solver import _check_part
from maesolvers.neural.samplers import Collocation

if TYPE_CHECKING:
    import torch

Activation = Literal["sigmoid", "tanh", "softplus"]
"""The smooth activations :class:`MLP` offers; ReLU has no second derivative."""


class Network(ABC):
    """Base class of every network part (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements :meth:`build`,
    and overrides :meth:`supports` if it does not handle every problem.
    """

    def supports(self, problem: Problem) -> bool:
        """Whether the network can represent the problem's solution; True by default.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return True

    @abstractmethod
    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Make the module ``u`` for one problem.

        Parameters
        ----------
        problem
            A problem the network supports.
        collocation
            The collocation points of the solve, for a network that fits
            something to the data before training.

        Returns
        -------
        torch.nn.Module
            Maps points of shape ``(M, d)`` to values of shape ``(M,)``;
            the parameters that require a gradient are the ones the solver
            trains.
        """


@dataclass(frozen=True, kw_only=True)
class MLP(Network):
    """A fully connected feed-forward network with a linear output layer.

    ``depth`` hidden layers of ``width`` units each, with the same smooth
    activation, then one linear output unit, so ``width = 10, depth = 5`` in
    2D has 481 parameters (Nystrom-Vestberg 2023, Example 4.1). Every weight
    and bias starts from the standard normal distribution, as in that paper
    (Section 4).

    Parameters
    ----------
    width
        Units per hidden layer, at least 1.
    depth
        Number of hidden layers, at least 1.
    activation
        ``"sigmoid"``, ``"tanh"`` or ``"softplus"``.

    Raises
    ------
    TypeError, ValueError
        On construction, for a size that is not an integer or is less than 1,
        or an unknown activation.
    """

    width: int = 10
    depth: int = 5
    activation: Activation = "sigmoid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", integer("width", self.width, 1))
        object.__setattr__(self, "depth", integer("depth", self.depth, 1))
        if self.activation not in ("sigmoid", "tanh", "softplus"):
            raise ValueError(
                "activation must be 'sigmoid', 'tanh' or 'softplus', "
                f"got {self.activation!r}"
            )

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Make the network for the problem's dimension.

        Parameters
        ----------
        problem
            Any problem.
        collocation
            Not used.

        Returns
        -------
        torch.nn.Module
        """
        from maesolvers.neural import _torch

        return _torch.mlp(problem.dim, self.width, self.depth, self.activation)


@dataclass(frozen=True, kw_only=True)
class HardDirichlet(Network):
    """The ansatz ``u = G + D N``, which meets the Dirichlet condition by construction.

    Nystrom-Vestberg 2023, eq. (3.1), after Berg and Nystrom 2018: ``N`` is
    the trained network; ``G`` is a network fitted to the boundary data
    ``g`` at the collocation boundary points; ``D`` is a network fitted to
    the distance to the boundary, at the first ``distance_points`` interior
    collocation points and at the boundary points (where it is 0). ``G`` and
    ``D`` have one hidden layer of tanh units, are fitted by L-BFGS on the
    mean squared error before training, and stay fixed during it. So ``u``
    meets the boundary condition as well as ``G`` fits ``g`` and ``D``
    vanishes there, whatever ``N`` is.

    Parameters
    ----------
    network
        The network part ``N``.
    lift_width
        Hidden units of ``G``, at least 1 (20 in the paper).
    distance_width
        Hidden units of ``D``, at least 1 (40 in the paper).
    distance_points
        Interior points at which ``D`` is fitted, at least 1 (300 in the
        paper); at most the number of interior collocation points are used.
    fit_steps
        Most L-BFGS iterations for each of the fits of ``G`` and ``D``, at
        least 1; each fit stops earlier when the largest component of its
        gradient is at most 1e-5, the default of the SciPy BFGS the paper
        used.

    Raises
    ------
    TypeError
        On construction, if ``network`` is not an instance of a well-defined
        :class:`Network`.
    TypeError, ValueError
        On construction, for a count that is not an integer or is less than 1.
    """

    network: Network
    lift_width: int = 20
    distance_width: int = 40
    distance_points: int = 300
    fit_steps: int = 1000

    def __post_init__(self) -> None:
        _check_part(f"{type(self).__name__}.network", self.network, Network)
        for name in ("lift_width", "distance_width", "distance_points", "fit_steps"):
            object.__setattr__(self, name, integer(name, getattr(self, name), 1))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is Dirichlet on a domain that knows its boundary distance.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return (
            isinstance(problem.boundary, Dirichlet)
            and isinstance(problem.domain, BoundaryDistance)
            and self.network.supports(problem)
        )

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Fit ``G`` and ``D``, then return ``G + D N`` with ``G`` and ``D`` frozen.

        Parameters
        ----------
        problem
            A problem the ansatz supports.
        collocation
            The collocation points; ``G`` and ``D`` are fitted at them.

        Returns
        -------
        torch.nn.Module
            Its trainable parameters are those of ``N``.

        Raises
        ------
        ValueError
            If the collocation set has no boundary points.
        """
        from maesolvers.neural import _torch

        if len(collocation.boundary) == 0:
            raise ValueError(
                "HardDirichlet fits G and D at the boundary points; the sampler "
                "gave none"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports checked it
        assert isinstance(problem.domain, BoundaryDistance)
        boundary = np.asarray(collocation.boundary)
        inner = np.asarray(collocation.interior[: self.distance_points])
        dim = problem.dim
        lift = _torch.mlp(dim, self.lift_width, 1, "tanh")
        _torch.fit(lift, boundary, problem.boundary.g(boundary), self.fit_steps)
        distance = _torch.mlp(dim, self.distance_width, 1, "tanh")
        _torch.fit(
            distance,
            np.concatenate([inner, boundary]),
            np.concatenate(
                [problem.domain.boundary_distance(inner), np.zeros(len(boundary))]
            ),
            self.fit_steps,
        )
        return _torch.Ansatz(lift, distance, self.network.build(problem, collocation))
