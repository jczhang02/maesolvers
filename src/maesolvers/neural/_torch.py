"""The torch side of the neural parts: modules, the least-squares fit, numpy wrappers.

Imported only inside methods that run during a solve, so that
``import maesolvers`` and the part classes need no torch (ADR 0006).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# torch's strong Wolfe line search, private to torch.optim.lbfgs: the line search
# of torch.optim.LBFGS, so both quasi-Newton parts search alike.
from torch.optim.lbfgs import _strong_wolfe

from maesolvers.neural import functional

_ACTIVATIONS = {
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "softplus": torch.nn.Softplus,
}

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


def fit(
    module: torch.nn.Module, x: np.ndarray, y: np.ndarray, max_steps: int, tol: float
) -> dict[str, float | int | bool]:
    """Fit ``module(x)`` to ``y`` by L-BFGS on the mean squared error, in place.

    Full batch, strong Wolfe line search; stops after ``max_steps``
    iterations or when the largest gradient component is at most ``tol``.

    Parameters
    ----------
    module
        Maps ``(M, d)`` to ``(M,)``; its parameters are changed.
    x, y
        Points of shape ``(M, d)`` and target values of shape ``(M,)``.
    max_steps
        Most L-BFGS iterations.
    tol
        Stop when the largest component of the gradient is at most this.

    Returns
    -------
    dict
        How the fit ended: ``iterations`` (L-BFGS iterations taken),
        ``max_gradient`` (largest gradient component at the end),
        ``converged`` (``max_gradient <= tol``), ``mse`` (the mean squared
        error) and ``max_misfit`` (the largest ``|module(x) - y|``).
    """
    dtype = next(module.parameters()).dtype
    points = torch.tensor(np.asarray(x), dtype=dtype)
    target = torch.tensor(np.asarray(y), dtype=dtype)
    parameters = list(module.parameters())
    optimizer = torch.optim.LBFGS(
        parameters,
        max_iter=max_steps,
        max_eval=2 * max_steps,
        tolerance_grad=tol,
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
    mse = float(closure())
    max_gradient = max(
        (float(p.grad.abs().max()) for p in parameters if p.grad is not None),
        default=0.0,
    )
    optimizer.zero_grad()
    with torch.no_grad():
        misfit = float((module(points) - target).abs().max())
    return {
        "iterations": int(optimizer.state[parameters[0]].get("n_iter", 0)),
        "max_gradient": max_gradient,
        "converged": max_gradient <= tol,
        "mse": mse,
        "max_misfit": misfit,
    }


class InputConvex(torch.nn.Module):
    """An input convex network (Amos et al. 2017; Nystrom-Vestberg 2023, eq. (3.7)).

    ``z_1 = s(L_0 x + b_0)``, ``z_{j+1} = s(W_j z_j + L_j x + b_j)`` for the
    later hidden layers, and the output ``W_k z_k + L_k x + b_k``, with
    ``s`` the softplus. Each ``W_j`` is the elementwise square of the trained
    parameter ``V_j``, so it is non-negative whatever ``V_j`` is; the
    ``L_j`` and ``b_j`` are free. Every parameter starts from the normal
    distribution with mean 0 and standard deviation ``std``.

    Parameters
    ----------
    dim
        Input dimension.
    width, depth
        Units per hidden layer and number of hidden layers.
    std
        Standard deviation of the initial parameters.
    """

    def __init__(self, dim: int, width: int, depth: int, std: float) -> None:
        super().__init__()
        self.inputs = torch.nn.ModuleList(
            [torch.nn.Linear(dim, width) for _ in range(depth)]
            + [torch.nn.Linear(dim, 1)]
        )
        self.roots = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(width, width)) for _ in range(depth - 1)]
            + [torch.nn.Parameter(torch.empty(1, width))]
        )
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(std=std)

    def weights(self) -> list[torch.Tensor]:
        """Return the hidden-to-hidden and hidden-to-output weights ``W_j = V_j^2``."""
        return [root.square() for root in self.roots]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, *later = self.inputs
        z = torch.nn.functional.softplus(first(x))
        weights = self.weights()
        for layer, (inputs, weight) in enumerate(zip(later, weights, strict=True)):
            z = z @ weight.T + inputs(x)
            if layer < len(later) - 1:
                z = torch.nn.functional.softplus(z)
        return z.squeeze(-1)


class DenseBFGS(torch.optim.Optimizer):
    """BFGS with a dense inverse Hessian approximation: one iteration per ``step``.

    The iteration of SciPy's ``minimize(method="BFGS")``. The approximation
    ``H`` starts as the identity; after each iteration, with ``s`` the
    change of the parameters and ``y`` that of the gradient,
    ``H <- (I - r s y') H (I - r y s') + r s s'``, ``r = 1 / y's``, skipped
    when ``y's <= 1e-10``, which keeps ``H`` positive definite. The step
    along ``d = -H g`` comes from torch's strong Wolfe line search (the one
    ``torch.optim.LBFGS`` uses), started, as SciPy starts its own, from
    ``min(1, 2.02 (f - f_prev) / g'd)``: the step that would repeat the last
    decrease of the loss, with ``f_prev = f + |g| / 2`` on the first
    iteration. ``H`` takes ``n^2`` numbers for ``n`` parameters.

    Parameters
    ----------
    params
        The parameters, in one group.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter]) -> None:
        super().__init__(params, {})
        if len(self.param_groups) != 1:
            raise ValueError("DenseBFGS takes one parameter group")

    @staticmethod
    def _flat_grad(params: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros_like(p).view(-1) if p.grad is None else p.grad.view(-1)
                for p in params
            ]
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        """Take one BFGS iteration.

        Parameters
        ----------
        closure
            Zeroes the gradients, evaluates the loss, back-propagates it and
            returns it.

        Returns
        -------
        torch.Tensor
            The loss at the parameters the step starts from.
        """
        evaluate = torch.enable_grad()(closure)
        params: list[torch.Tensor] = self.param_groups[0]["params"]
        state = self.state[params[0]]
        loss = evaluate()
        f, g = float(loss), self._flat_grad(params)
        if not g.any():  # a stationary point: no direction to search
            return loss
        x = parameters_to_vector(params).clone()
        if "h" not in state:
            state["h"] = torch.eye(len(x), dtype=x.dtype, device=x.device)
            state["f"] = f + float(g.norm()) / 2
        else:
            s, y = x - state["x"], g - state["g"]
            ys = float(y.dot(s))
            if ys > 1e-10:
                h, r = state["h"], 1.0 / ys
                hy = h @ y
                h += (r * r * float(y.dot(hy)) + r) * torch.outer(s, s)
                h -= r * (torch.outer(hy, s) + torch.outer(s, hy))
        d = -(state["h"] @ g)
        gtd = g.dot(d)
        t = min(1.0, 2.02 * (f - state["f"]) / float(gtd))
        if t <= 0.0:  # the loss rose in the last step: no scale to repeat
            t = 1.0
        state["x"], state["g"], state["f"] = x, g, f

        def directional(
            start: torch.Tensor, step: float, direction: torch.Tensor
        ) -> tuple[float, torch.Tensor]:
            vector_to_parameters(start + step * direction, params)
            value = float(evaluate())
            gradient = self._flat_grad(params)
            vector_to_parameters(start, params)
            return value, gradient

        _, _, t, _ = _strong_wolfe(directional, x, t, d, f, g, gtd)
        vector_to_parameters(x + t * d, params)
        return loss


class Ansatz(torch.nn.Module):
    """``u = G + D N`` with ``G`` and ``D`` frozen: the module of HardDirichlet.

    Parameters
    ----------
    lift, distance, network
        ``G``, ``D`` and ``N``, each mapping ``(M, d)`` to ``(M,)``.
    report
        How the fits of ``G`` and ``D`` ended; the solver records it as
        ``Result.info["network"]``.
    """

    def __init__(
        self,
        lift: torch.nn.Module,
        distance: torch.nn.Module,
        network: torch.nn.Module,
        report: dict,
    ) -> None:
        super().__init__()
        self.lift = lift.requires_grad_(False)
        self.distance = distance.requires_grad_(False)
        self.network = network
        self.report = report

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lift(x) + self.distance(x) * self.network(x)


@contextmanager
def building(seed: int) -> Iterator[None]:
    """Build modules in float64 from a seeded generator, leaving torch's state as it was.

    Parameters
    ----------
    seed
        Seeds torch's CPU generator inside the block; the generator's state
        outside is restored afterwards. The CUDA generators are not touched
        (``torch.manual_seed`` would seed them too), since the modules are
        built on the CPU.
    """
    default = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
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
