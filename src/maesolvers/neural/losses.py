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

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from maesolvers.core._params import boolean, nonnegative, positive
from maesolvers.core.boundary import Dirichlet, Transport
from maesolvers.core.domains import Domain
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
    target
        The target set of a
        :class:`~maesolvers.core.boundary.Transport` problem, for a loss on
        the transport condition (with
        :func:`maesolvers.neural.functional.signed_distance`); None for
        another boundary condition. Not a tensor: :meth:`to` leaves it as it
        is.
    target_boundary
        Shape ``(J, d)``: points on the boundary of the target set, for a
        loss that compares the image of the boundary with it
        (:class:`HausdorffLoss`); None when the sampler drew none.
    """

    interior: "torch.Tensor"
    f: "torch.Tensor"
    boundary: "torch.Tensor"
    g: "torch.Tensor | None"
    f_boundary: "torch.Tensor"
    target: "Domain | None" = None
    target_boundary: "torch.Tensor | None" = None

    def to(self, device: "torch.device | str") -> "Batch":
        """Return the batch with every tensor on ``device``; the other fields stay.

        Parameters
        ----------
        device
            The device, as torch takes it: ``"cpu"``, ``"cuda:0"``, a
            ``torch.device``.

        Returns
        -------
        Batch
        """
        import torch

        moved = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                moved[field.name] = value.to(device)
        return dataclasses.replace(self, **moved)


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
        ``batch.g``, which is None there; a loss for
        :class:`~maesolvers.core.boundary.Transport` reads ``batch.target``.

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
        object.__setattr__(
            self,
            "equation_at_boundary",
            boolean("equation_at_boundary", self.equation_at_boundary),
        )

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


@dataclass(frozen=True, kw_only=True)
class HausdorffLoss(Loss):
    """The equation residual plus a Hausdorff-type distance between ``grad u(dX)`` and ``dY``.

    The loss of Caboussat-Peruso 2025 (Section 3.3) for a
    :class:`~maesolvers.core.boundary.Transport` problem, the map ``grad u``
    of the domain ``X`` onto the target set ``Y``::

        E = (1/M) sum_j (det D^2 u(x_j) - f(x_j))^2
            + C [ (1/K) sum_k min_l |grad u(y_k) - t_l|^2
                + (1/L) sum_l min_k |grad u(y_k) - t_l|^2 ]

    over the interior points ``x_j``, the boundary points ``y_k`` of ``X``
    and the points ``t_l`` on the boundary of ``Y``, with
    ``C = boundary_weight``. The bracket is eq. (3.4), the discrete form of
    the condition ``grad u(dX) = dY`` through the Hausdorff distance: its
    first sum keeps the image of each boundary point near ``dY`` (the
    paper's injectivity term), its second leaves no point of ``dY`` far
    from the image (surjectivity). The paper's equation
    ``det D^2 u = f / g(grad u)`` has the target density ``g``; a
    ``Transport`` problem's target density is uniform 1, so the residual is
    ``det D^2 u - f``. Both terms are means, as eq. (3.4) and the
    equation term ``E_PDE`` of the paper print them. The terms are logged
    as ``equation``, ``into`` (the first sum) and ``onto`` (the second);
    the squared distances come from
    :func:`~maesolvers.neural.functional.nearest_squared_distances`.

    Parameters
    ----------
    boundary_weight
        ``C``, positive (1 in the paper): with no other boundary condition,
        this term alone imposes the transport condition.

    Raises
    ------
    TypeError, ValueError
        On construction, for a weight that is not a positive real.
    ValueError
        When called on a batch without boundary points or without target
        boundary points (the sampler must draw both, as
        :class:`~maesolvers.neural.samplers.TransportSampler` does), so at
        the first training step.
    """

    boundary_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "boundary_weight", positive("boundary_weight", self.boundary_weight)
        )

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is a transport problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return isinstance(problem.boundary, Transport)

    def __call__(self, u: "torch.nn.Module", batch: Batch) -> "dict[str, torch.Tensor]":
        """Return ``loss``, ``equation``, ``into`` and ``onto``.

        Parameters
        ----------
        u
            The module being trained.
        batch
            The collocation points, the data there and the target boundary
            points.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If the batch has no boundary points or no target boundary points.
        """
        from maesolvers.neural import functional

        target = batch.target_boundary
        if batch.boundary.shape[0] == 0 or target is None:
            raise ValueError(
                "HausdorffLoss needs boundary points and target boundary points; "
                "use a sampler that draws both, such as TransportSampler"
            )
        hess = functional.hessian(u, batch.interior)
        image = functional.gradient(u, batch.boundary)
        into, onto = functional.nearest_squared_distances(image, target)
        terms = {
            "equation": functional.monge_ampere_residual(hess, batch.f).pow(2).mean(),
            "into": into.mean(),
            "onto": onto.mean(),
        }
        loss = terms["equation"] + self.boundary_weight * (
            terms["into"] + terms["onto"]
        )
        return {"loss": loss, **terms}
