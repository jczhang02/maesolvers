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

import numpy as np

from maesolvers.core._params import nonnegative
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.problem import Problem

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Batch:
    """The collocation points of a solve and the problem's data there, as tensors.

    All in float64 and on one device: the training device during training,
    the CPU when the solver evaluates the trained network's loss terms.

    Parameters
    ----------
    interior
        Shape ``(M, d)``: the interior collocation points.
    f
        Shape ``(M,)``: the right-hand side at ``interior``.
    boundary
        Shape ``(K, d)``: the boundary collocation points.
    g
        Shape ``(K,)``: the Dirichlet data at ``boundary``; None for a
        problem with another boundary condition, so that a loss that reads
        it there fails instead of fitting made-up data.
    f_boundary
        Shape ``(K,)``: the right-hand side at ``boundary``, for a loss that
        also evaluates the equation there.
    """

    interior: "torch.Tensor"
    f: "torch.Tensor"
    boundary: "torch.Tensor"
    g: "torch.Tensor | None"
    f_boundary: "torch.Tensor"


class Loss(ABC):
    """Base class of every loss part (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its weights, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements
    :meth:`__call__`, and overrides :meth:`supports` if it does not handle
    every Dirichlet problem, or if it handles another boundary condition.
    """

    def supports(self, problem: Problem) -> bool:
        """Whether the loss is a valid objective for the problem.

        A :class:`Batch` carries Dirichlet data only, so by default a loss
        supports exactly the problems with a
        :class:`~maesolvers.core.boundary.Dirichlet` boundary condition. A
        loss for another condition overrides this, and must not read
        ``batch.g``, which is None there.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return isinstance(problem.boundary, Dirichlet)

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
    """The equation residual plus penalties on non-convexity and on the boundary misfit.

    The loss of Nystrom-Vestberg 2023 (Section 3) is

        E = sum_j (det D^2 u(x_j) - f(x_j))^2
            + C (1/M) sum_j (min(0, u_xx(x_j))^2 + min(0, u_yy(x_j))^2)
            + B (1/K) sum_k (u(y_k) - g(y_k))^2

    over the interior points ``x_j`` and the boundary points ``y_k``, with
    ``C = convexity_weight`` and ``B = boundary_weight``. The penalty MLP
    (eqs. (3.3) to (3.5)) has ``C > 0`` and ``B = 0``, its ansatz meeting
    the boundary condition by itself; the input convex network (eq. (3.8))
    has ``C = 0``, being convex by construction, and ``B > 0``. The
    equation term is a sum, not a mean, as the paper prints it. With
    ``equation_at_boundary`` the sum also runs over the boundary points
    ``y_k``, as in the paper's Example 4.2; the convexity term stays a mean
    over the interior points. The terms are logged as ``equation``,
    ``convexity`` and, when ``B > 0``, ``boundary``.

    Parameters
    ----------
    convexity_weight
        ``C``, at least 0 (``1e4`` for the penalty MLP); with 0 no penalty.
    boundary_weight
        ``B``, at least 0 (100 for the input convex network); with 0 no
        boundary term. With ``B > 0`` the sampler must give boundary points.
    equation_at_boundary
        Whether the equation term also sums over the boundary points. Then
        the sampler must give boundary points.

    Raises
    ------
    TypeError, ValueError
        On construction, for a weight that is not a real number, negative or
        not finite, or an ``equation_at_boundary`` that is not a bool.
    ValueError
        When called with ``B > 0`` or ``equation_at_boundary`` on a batch
        without boundary points, so at the first training step, before any
        update, or with ``B > 0`` without Dirichlet data.
    """

    convexity_weight: float = 1e4
    boundary_weight: float = 0.0
    equation_at_boundary: bool = False

    def __post_init__(self) -> None:
        for name in ("convexity_weight", "boundary_weight"):
            object.__setattr__(self, name, nonnegative(name, getattr(self, name)))
        flag = self.equation_at_boundary
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(
                f"equation_at_boundary must be a bool, got {type(flag).__name__}"
            )
        object.__setattr__(self, "equation_at_boundary", bool(flag))

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
        return super().supports(problem) and (
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

        Raises
        ------
        ValueError
            If ``boundary_weight > 0`` or ``equation_at_boundary`` and the
            batch has no boundary points, or if ``boundary_weight > 0`` and
            the batch has no Dirichlet data (``batch.g`` is None).
        """
        import torch

        from maesolvers.neural import functional

        for flag, wanted in (
            ("boundary_weight > 0", self.boundary_weight > 0.0),
            ("equation_at_boundary", self.equation_at_boundary),
        ):
            if wanted and batch.boundary.shape[0] == 0:
                raise ValueError(
                    f"PenaltyLoss has {flag} but the sampler gave no boundary points"
                )
        points, f = batch.interior, batch.f
        if self.equation_at_boundary:
            points = torch.cat([points, batch.boundary])
            f = torch.cat([f, batch.f_boundary])
        hess = functional.hessian(u, points)
        interior = hess[: batch.interior.shape[0]]
        terms = {
            "equation": functional.monge_ampere_residual(hess, f).pow(2).sum(),
            "convexity": functional.diagonal_convexity_penalty(interior),
        }
        loss = terms["equation"] + self.convexity_weight * terms["convexity"]
        if self.boundary_weight > 0.0:
            if batch.g is None:
                raise ValueError(
                    "PenaltyLoss has boundary_weight > 0 but the batch has no "
                    "Dirichlet data"
                )
            terms["boundary"] = functional.mean_squared_misfit(
                u(batch.boundary), batch.g
            )
            loss = loss + self.boundary_weight * terms["boundary"]
        return {"loss": loss, **terms}
