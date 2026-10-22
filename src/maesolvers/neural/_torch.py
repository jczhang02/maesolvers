"""The torch side of the neural parts: modules, the least-squares fit, numpy wrappers.

Imported only inside methods that run during a solve, so that
``import maesolvers`` and the part classes need no torch (ADR 0006).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from maesolvers.neural import functional

_ACTIVATIONS = {
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "softplus": torch.nn.Softplus,
}

_FIT_TOLERANCE = 1e-5
"""Largest gradient component at which a fit stops: SciPy's BFGS default ``gtol``."""

_CHUNK = 16384
"""Points per forward pass when a result is evaluated."""


class _Squeeze(torch.nn.Module):
    """Drop the last axis, of length 1: values of shape ``(M,)``, not ``(M, 1)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)


def mlp(dim: int, width: int, depth: int, activation: str) -> torch.nn.Module:
    """Make a feed-forward network with standard normal initial weights and biases.

    Parameters
    ----------
    dim
        Input dimension.
    width, depth
        Units per hidden layer and number of hidden layers.
    activation
        A key of the activations: ``"sigmoid"``, ``"tanh"``, ``"softplus"``.

    Returns
    -------
    torch.nn.Module
        Maps ``(M, dim)`` to ``(M,)``.
    """
    sizes = [dim] + [width] * depth
    layers: list[torch.nn.Module] = []
    for fan_in, fan_out in itertools.pairwise(sizes):
        layers += [torch.nn.Linear(fan_in, fan_out), _ACTIVATIONS[activation]()]
    layers += [torch.nn.Linear(width, 1), _Squeeze()]
    network = torch.nn.Sequential(*layers)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_()
    return network


def fit(module: torch.nn.Module, x: np.ndarray, y: np.ndarray, max_steps: int) -> float:
    """Fit ``module(x)`` to ``y`` by L-BFGS on the mean squared error, in place.

    Full batch, strong Wolfe line search; stops after ``max_steps``
    iterations or when the largest gradient component is at most 1e-5.

    Parameters
    ----------
    module
        Maps ``(M, d)`` to ``(M,)``; its parameters are changed.
    x, y
        Points of shape ``(M, d)`` and target values of shape ``(M,)``.
    max_steps
        Most L-BFGS iterations.

    Returns
    -------
    float
        The final mean squared error.
    """
    dtype = next(module.parameters()).dtype
    points = torch.tensor(np.asarray(x), dtype=dtype)
    target = torch.tensor(np.asarray(y), dtype=dtype)
    optimizer = torch.optim.LBFGS(
        module.parameters(),
        max_iter=max_steps,
        max_eval=2 * max_steps,
        tolerance_grad=_FIT_TOLERANCE,
        tolerance_change=0.0,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = functional.mean_squared_misfit(module(points), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return float(functional.mean_squared_misfit(module(points), target))


class Ansatz(torch.nn.Module):
    """``u = G + D N`` with ``G`` and ``D`` frozen: the module of HardDirichlet.

    Parameters
    ----------
    lift, distance, network
        ``G``, ``D`` and ``N``, each mapping ``(M, d)`` to ``(M,)``.
    """

    def __init__(
        self,
        lift: torch.nn.Module,
        distance: torch.nn.Module,
        network: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.lift = lift.requires_grad_(False)
        self.distance = distance.requires_grad_(False)
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lift(x) + self.distance(x) * self.network(x)


@contextmanager
def building(seed: int) -> Iterator[None]:
    """Build modules in float64 from a seeded generator, leaving torch's state as it was.

    Parameters
    ----------
    seed
        Seeds torch's CPU generator inside the block; the generator's state
        outside is restored afterwards.
    """
    default = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float64)
        try:
            yield
        finally:
            torch.set_default_dtype(default)


def trainable(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return the parameters of ``module`` that require a gradient."""
    return [p for p in module.parameters() if p.requires_grad]


def _chunks(points: np.ndarray, dim: int) -> tuple[Iterator[torch.Tensor], tuple]:
    """Flat float64 tensors of at most ``_CHUNK`` points, and the leading shape."""
    x = np.asarray(points, dtype=float)
    if x.ndim == 0 or x.shape[-1] != dim:
        raise ValueError(f"points must have shape (..., {dim}), got {x.shape}")
    flat = torch.tensor(x.reshape(-1, dim), dtype=torch.float64)
    return iter(torch.split(flat, _CHUNK)), x.shape[:-1]


def values(module: torch.nn.Module, points: np.ndarray, dim: int) -> np.ndarray:
    """Evaluate ``module`` at points of shape ``(..., dim)``; float64 of shape ``(...)``."""
    chunks, shape = _chunks(points, dim)
    with torch.no_grad():
        out = [module(chunk) for chunk in chunks]
    return torch.cat(out).numpy().reshape(shape) if out else np.empty(shape)


def hessians(module: torch.nn.Module, points: np.ndarray, dim: int) -> np.ndarray:
    """Return the Hessian of ``module`` at points ``(..., dim)``: ``(..., dim, dim)``."""
    chunks, shape = _chunks(points, dim)
    out = [
        functional.hessian(module, chunk, create_graph=False).detach()
        for chunk in chunks
    ]
    if not out:
        return np.empty((*shape, dim, dim))
    return torch.cat(out).numpy().reshape(*shape, dim, dim)
