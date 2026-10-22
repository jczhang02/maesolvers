"""Pure torch kernels of the neural solvers: derivatives and loss terms.

Every function here takes tensors and returns tensors, with no state, so a
loss of your own builds on them and a test checks them without a solver
(ADR 0004). Points are tensors of shape ``(M, d)`` and a network maps them to
values of shape ``(M,)``. This module imports torch.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Function = Callable[[torch.Tensor], torch.Tensor]
"""Maps points of shape ``(M, d)`` to values of shape ``(M,)``, such as a network."""


def hessian(u: Function, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Hessian of ``u`` at each point, by automatic differentiation.

    Parameters
    ----------
    u
        A twice differentiable function of the points, such as a network.
    x
        Points of shape ``(M, d)``; not changed.
    create_graph
        Keep the graph of the derivatives, so that a loss built on the
        Hessian can be differentiated with respect to the network's
        parameters. False for evaluation only.

    Returns
    -------
    torch.Tensor
        Shape ``(M, d, d)``, row ``i`` the gradient of ``du/dx_i``.
    """
    x = x.detach().requires_grad_(True)
    m, d = x.shape
    with torch.enable_grad():
        (gradient,) = torch.autograd.grad(
            u(x).sum(), x, create_graph=True, allow_unused=True
        )
        if gradient is None or not gradient.requires_grad:  # u is affine in x
            return x.new_zeros(m, d, d)
        rows = [
            torch.autograd.grad(
                gradient[:, i].sum(),
                x,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
            for i in range(d)
        ]
    return torch.stack(
        [x.new_zeros(m, d) if row is None else row for row in rows], dim=1
    )


def monge_ampere_residual(hess: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Pointwise residual ``det D^2 u - f``.

    Parameters
    ----------
    hess
        Hessians, shape ``(M, d, d)``.
    f
        Right-hand side at the same points, shape ``(M,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)``.
    """
    return torch.linalg.det(hess) - f


def diagonal_convexity_penalty(hess: torch.Tensor) -> torch.Tensor:
    """Mean over the points of ``sum_i min(0, u_{x_i x_i})^2``.

    Nystrom-Vestberg 2023, eq. (3.4). In 2D a symmetric matrix with a
    nonnegative determinant is positive semidefinite exactly when both
    diagonal entries are nonnegative, so together with the equation term
    the penalty pushes towards a convex solution; in higher dimensions it
    does not.

    Parameters
    ----------
    hess
        Hessians, shape ``(M, d, d)``.

    Returns
    -------
    torch.Tensor
        A scalar, zero when every diagonal entry is nonnegative.
    """
    diagonal = torch.diagonal(hess, dim1=-2, dim2=-1)
    return torch.clamp(diagonal, max=0.0).pow(2).sum(dim=-1).mean()


def mean_squared_misfit(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of ``(values - target)^2``, zero for empty tensors.

    Parameters
    ----------
    values, target
        Tensors of the same shape, such as ``u`` and ``g`` at boundary points.

    Returns
    -------
    torch.Tensor
        A scalar.
    """
    if values.numel() == 0:
        return values.new_zeros(())
    return (values - target).pow(2).mean()
