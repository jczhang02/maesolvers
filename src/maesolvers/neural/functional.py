"""Pure torch kernels of the neural solvers: derivatives and loss terms.

Every function here takes tensors (and, for :func:`signed_distance`, the
target set) and returns tensors, with no state, so a loss of your own builds
on them and a test checks them without a solver (ADR 0004). Points are
tensors of shape ``(M, d)`` and a network maps them to values of shape
``(M,)``. This module imports torch.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from maesolvers.core.domains import Domain
from maesolvers.core.transport import closest_boundary_point

Function = Callable[[torch.Tensor], torch.Tensor]
"""Maps points of shape ``(M, d)`` to values of shape ``(M,)``, such as a network."""


def gradient(u: Function, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Gradient of ``u`` at each point, by automatic differentiation.

    The transport map ``grad u`` of a transport problem, for a loss on the
    transport condition or for :func:`maesolvers.metrics.gradient_map_error`.

    Parameters
    ----------
    u
        A differentiable function of the points, such as a network.
    x
        Points of shape ``(M, d)``; not changed.
    create_graph
        Keep the graph, so that a loss built on the gradient can be
        differentiated with respect to the network's parameters. False for
        evaluation only.

    Returns
    -------
    torch.Tensor
        Shape ``(M, d)``; zero for a ``u`` that depends on its parameters
        but not on ``x``.
    """
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        (grad,) = torch.autograd.grad(
            u(x).sum(), x, create_graph=create_graph, allow_unused=True
        )
    return torch.zeros_like(x) if grad is None else grad


def signed_distance(target: Domain, y: torch.Tensor) -> torch.Tensor:
    """Signed distance to the target's boundary, differentiable in ``y``.

    The torch form of :func:`maesolvers.core.transport.signed_distance`, with
    its sign (positive outside the target) and its targets (``Box``,
    ``Disk``, ``Ellipse``), for a loss on the transport condition
    ``signed_distance(target, grad u) = 0`` on the boundary. The nearest
    boundary point ``p`` and the outward unit normal ``nu`` there come from
    the numpy function, outside the graph, and the result is
    ``(y - p) . nu``: its value is the signed distance, and its gradient in
    ``y`` is ``nu``, the gradient of the signed distance wherever the
    nearest point is unique, so a loss trains through it with no NaN on the
    boundary. Its second derivatives in ``y`` are zero, not the curvature.

    Each call copies ``y`` to the CPU and the nearest points back, so on a
    GPU it waits for the device once per loss evaluation; the ellipse takes
    up to 64 bisection steps in numpy. Torch closed forms for ``Box`` and
    ``Disk``, which would avoid the copy, are left to the first loss that
    needs the speed.

    Parameters
    ----------
    target
        A :class:`~maesolvers.core.domains.Box`,
        :class:`~maesolvers.core.domains.Disk` or
        :class:`~maesolvers.core.domains.Ellipse`.
    y
        Points of shape ``(M, d)``, on any device.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)``, on the device and in the dtype of ``y``.

    Raises
    ------
    TypeError
        For another kind of target.
    """
    foot, normal = closest_boundary_point(target, y.detach().cpu().double().numpy())
    foot_t = torch.as_tensor(foot, dtype=y.dtype, device=y.device)
    normal_t = torch.as_tensor(normal, dtype=y.dtype, device=y.device)
    return ((y - foot_t) * normal_t).sum(dim=-1)


def nearest_squared_distances(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance from each point of one set to the nearest point of the other.

    The two halves of the discrete Hausdorff-type distance of
    Caboussat-Peruso 2025, eq. (3.4): ``min_j |a_i - b_j|^2`` for each
    ``a_i`` and ``min_i |a_i - b_j|^2`` for each ``b_j``. The squares come
    from the differences themselves, with no square root and no expansion
    ``|a|^2 + |b|^2 - 2 a . b``, so they are never negative and their
    gradient is finite where two points coincide; where several points tie
    for the nearest, the gradient goes to one of them. The work and memory
    grow as ``M K d``.

    Parameters
    ----------
    a
        Points of shape ``(M, d)``, ``M >= 1``.
    b
        Points of shape ``(K, d)``, ``K >= 1``, on the device and in the
        dtype of ``a``.

    Returns
    -------
    tuple of torch.Tensor
        Shapes ``(M,)`` and ``(K,)``.

    Raises
    ------
    ValueError
        If a set is empty: the distance to no points is not defined.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError(
            "nearest_squared_distances needs two non-empty sets, "
            f"got {a.shape[0]} and {b.shape[0]} points"
        )
    squared = (a[:, None, :] - b[None, :, :]).pow(2).sum(dim=-1)
    return squared.min(dim=1).values, squared.min(dim=0).values


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


def smooth_scaling(z: torch.Tensor, epsilon: float) -> torch.Tensor:
    """``epsilon Z / sqrt(1 + |Z|_F^2)`` for each matrix ``Z``: a smooth map into a ball.

    The smooth norm constraint of the QS-PINN note (Section 6, eq. (11)): it
    maps every matrix into the open Frobenius ball of radius ``epsilon``,
    so ``|P(Z)|_op <= |P(Z)|_F < epsilon``, and it is infinitely
    differentiable and ``epsilon``-Lipschitz, where the hard projection
    ``epsilon Z / max(1, |Z|)`` is not differentiable on the sphere.

    Parameters
    ----------
    z
        Matrices of shape ``(..., n, n)``.
    epsilon
        Radius of the ball, positive.

    Returns
    -------
    torch.Tensor
        The shape of ``z``.
    """
    norm2 = z.pow(2).sum(dim=(-2, -1), keepdim=True)
    return epsilon * z / torch.sqrt(1.0 + norm2)


def quadratic_form(
    matrix: torch.Tensor,
    x: torch.Tensor,
    reference: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """``(x - x0)^T A^T A (x - x0) + b . x + c`` at each point: the QS form.

    The solution of QS-PINN (note, Definition 3.1) for a matrix ``A`` per
    point. Whatever ``A`` is, the quadratic term is ``|A (x - x0)|^2 >= 0``,
    so ``u`` lies above its tangent plane ``c + b . x`` at ``x0``, where its
    gradient is ``b`` and its Hessian ``2 A^T A`` when ``A`` is smooth
    (Proposition 3.3).

    Parameters
    ----------
    matrix
        ``A`` at the points, shape ``(M, d, d)``.
    x
        Points of shape ``(M, d)``.
    reference
        The reference point ``x0``, shape ``(d,)``.
    b
        The constant gradient at ``x0``, shape ``(d,)``.
    c
        The constant term, a scalar tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)``.
    """
    image = (matrix @ (x - reference).unsqueeze(-1)).squeeze(-1)
    return image.pow(2).sum(dim=-1) + x @ b + c


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
    does not. The equation term keeps the determinant near ``f``, so where
    ``f = 0`` (T4, T5, T20) a saddle with a slightly negative determinant
    and a nonnegative diagonal is not penalised.

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
    """Mean of ``(values - target)^2``.

    Parameters
    ----------
    values, target
        Tensors of the same shape, such as ``u`` and ``g`` at boundary points.

    Returns
    -------
    torch.Tensor
        A scalar.

    Raises
    ------
    ValueError
        For empty tensors: a misfit at no points is no constraint, so a loss
        term built on it would drop out silently.
    """
    if values.numel() == 0:
        raise ValueError("mean_squared_misfit of empty tensors: there are no points")
    return (values - target).pow(2).mean()
