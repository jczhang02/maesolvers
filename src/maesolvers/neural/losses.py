"""Losses: the part of a neural solver that the training minimises.

A loss part (ADR 0004) is a frozen dataclass of weights whose
:meth:`Loss.__call__` returns the loss terms of a module on a :class:`Batch`,
the collocation points with the data there. The terms are built from the
pure kernels of :mod:`maesolvers.neural.functional`.

This module imports torch only inside ``__call__``, so the parts can be
configured, compared and recorded without the ``neural`` extra.
"""

# No `from __future__ import annotations` in this module: jsonargparse reads the
# field types of the parts (ADR 0005); see maesolvers.classical.iterations.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from maesolvers.core._params import nonnegative
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.problem import Problem

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Batch:
    """The collocation points of a solve and the problem's data there, as tensors.

    All on the training device, in float64.

    Parameters
    ----------
    interior
        Shape ``(M, d)``: the interior collocation points.
    f
        Shape ``(M,)``: the right-hand side at ``interior``.
    boundary
        Shape ``(K, d)``: the boundary collocation points.
    g
        Shape ``(K,)``: the Dirichlet data at ``boundary``; zeros for a
        problem with another boundary condition.
    """

    interior: "torch.Tensor"
    f: "torch.Tensor"
    boundary: "torch.Tensor"
    g: "torch.Tensor"


class Loss(ABC):
    """Base class of every loss part (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its weights, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements
    :meth:`__call__`, and overrides :meth:`supports` if it does not handle
    every problem.
    """

    def supports(self, problem: Problem) -> bool:
        """Whether the loss is a valid objective for the problem; True by default.

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
    def __call__(self, u: "torch.nn.Module", batch: Batch) -> "dict[str, torch.Tensor]":
        """Return the loss terms of ``u`` on the batch.

        Parameters
        ----------
        u
            The module being trained, mapping ``(M, d)`` to ``(M,)``.
        batch
            The collocation points and the data there.

        Returns
        -------
        dict
            Scalar tensors by name. ``"loss"`` is what the training minimises;
            every term is logged after each step under its name (ADR 0008).
        """


@dataclass(frozen=True, kw_only=True)
class PenaltyLoss(Loss):
    """The equation residual plus a penalty on non-convexity (Nystrom-Vestberg 2023, 3.1).

    The loss is

        E = sum_j (det D^2 u(x_j) - f(x_j))^2
            + C (1/M) sum_j (min(0, u_xx(x_j))^2 + min(0, u_yy(x_j))^2)
            + B (1/K) sum_k (u(y_k) - g(y_k))^2

    over the interior points ``x_j`` and the boundary points ``y_k``, with
    ``C = convexity_weight`` and ``B = boundary_weight``: eqs. (3.3) to (3.5)
    of the paper, whose equation term is a sum, not a mean, and a boundary
    term for an ansatz that does not meet the boundary condition by itself
    (the paper's ansatz does, so ``B = 0`` there). The terms are logged as
    ``equation``, ``convexity`` and, when ``B > 0``, ``boundary``.

    Parameters
    ----------
    convexity_weight
        ``C``, at least 0 (``1e4`` in the paper); with 0 no penalty.
    boundary_weight
        ``B``, at least 0; with 0 no boundary term.

    Raises
    ------
    TypeError, ValueError
        On construction, for a weight that is not a real number, negative or
        not finite.
    """

    convexity_weight: float = 1e4
    boundary_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("convexity_weight", "boundary_weight"):
            object.__setattr__(self, name, nonnegative(name, getattr(self, name)))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is Dirichlet, and 2D when the penalty is on.

        The diagonal penalty implies convexity only in 2D (see
        :func:`~maesolvers.neural.functional.diagonal_convexity_penalty`).

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return isinstance(problem.boundary, Dirichlet) and (
            self.convexity_weight == 0.0 or problem.dim == 2
        )

    def __call__(self, u: "torch.nn.Module", batch: Batch) -> "dict[str, torch.Tensor]":
        """Return ``loss``, ``equation``, ``convexity`` and, if weighted, ``boundary``.

        Parameters
        ----------
        u
            The module being trained.
        batch
            The collocation points and the data there.

        Returns
        -------
        dict
        """
        from maesolvers.neural import functional

        hess = functional.hessian(u, batch.interior)
        terms = {
            "equation": functional.monge_ampere_residual(hess, batch.f).pow(2).sum(),
            "convexity": functional.diagonal_convexity_penalty(hess),
        }
        loss = terms["equation"] + self.convexity_weight * terms["convexity"]
        if self.boundary_weight > 0.0:
            terms["boundary"] = functional.mean_squared_misfit(
                u(batch.boundary), batch.g
            )
            loss = loss + self.boundary_weight * terms["boundary"]
        return {"loss": loss, **terms}
