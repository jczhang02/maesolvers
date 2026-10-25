"""The torch side of the neural parts: modules, the least-squares fit, numpy wrappers.

Imported only inside methods that run during a solve, so that
``import maesolvers`` and the part classes need no torch (ADR 0006).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch

from maesolvers.neural import functional

_ACTIVATIONS = {
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "softplus": torch.nn.Softplus,
}

_CHUNK = 16384
"""Points per forward pass when a result is evaluated."""


class _Squeeze(torch.nn.Module):
    """Drop the last axis if it has length 1: values of shape ``(M,)``, not ``(M, 1)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)


def mlp(
    dim: int, width: int, depth: int, activation: str, outputs: int = 1
) -> torch.nn.Sequential:
    """Make a feed-forward network with standard normal initial weights and biases.

    Parameters
    ----------
    dim
        Input dimension.
    width, depth
        Units per hidden layer and number of hidden layers.
    activation
        A key of the activations: ``"sigmoid"``, ``"tanh"``, ``"softplus"``.
    outputs
        Output units of the linear output layer.

    Returns
    -------
    torch.nn.Sequential
        Maps ``(M, dim)`` to ``(M,)`` for one output, to ``(M, outputs)``
        for more; its last but one module is the linear output layer.
    """
    sizes = [dim] + [width] * depth
    layers: list[torch.nn.Module] = []
    for fan_in, fan_out in itertools.pairwise(sizes):
        layers += [torch.nn.Linear(fan_in, fan_out), _ACTIVATIONS[activation]()]
    layers += [torch.nn.Linear(width, outputs), _Squeeze()]
    network = torch.nn.Sequential(*layers)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_()
    return network


def _minimise(
    module: torch.nn.Module,
    misfit: Callable[[], torch.Tensor],
    max_steps: int,
    tol: float,
) -> dict[str, float | int | bool]:
    """Minimise the mean square of ``misfit()`` over the module's parameters by L-BFGS.

    Full batch, strong Wolfe line search, 100 curvature pairs; stops after
    ``max_steps`` iterations or when the largest gradient component is at
    most ``tol``. Returns what :func:`fit` reports.
    """
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
        loss = misfit().pow(2).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    mse = float(closure())
    max_gradient = max(
        (float(p.grad.abs().max()) for p in parameters if p.grad is not None),
        default=0.0,
    )
    optimizer.zero_grad()
    largest = float(misfit().detach().abs().max())
    return {
        "iterations": int(optimizer.state[parameters[0]].get("n_iter", 0)),
        "max_gradient": max_gradient,
        "converged": max_gradient <= tol,
        "mse": mse,
        "max_misfit": largest,
    }


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
    return _minimise(module, lambda: module(points) - target, max_steps, tol)


def fit_gradient(
    module: torch.nn.Module,
    x: np.ndarray,
    gradient: np.ndarray,
    max_steps: int,
    tol: float,
) -> dict[str, float | int | bool]:
    """Fit the gradient of ``module`` at ``x`` to ``gradient`` by L-BFGS, in place.

    The objective is the mean over the points and the components of
    ``(d module / dx_i - gradient_i)^2``; otherwise as :func:`fit`.

    Parameters
    ----------
    module
        Maps ``(M, d)`` to ``(M,)``; its parameters are changed.
    x, gradient
        Points and target gradients, both of shape ``(M, d)``.
    max_steps
        Most L-BFGS iterations.
    tol
        Stop when the largest component of the gradient of the objective is
        at most this.

    Returns
    -------
    dict
        As :func:`fit`, ``max_misfit`` the largest misfit of one component.
    """
    dtype = next(module.parameters()).dtype
    points = torch.tensor(np.asarray(x), dtype=dtype)
    target = torch.tensor(np.asarray(gradient), dtype=dtype)
    return _minimise(
        module,
        lambda: functional.gradient(module, points) - target,
        max_steps,
        tol,
    )


class InputConvex(torch.nn.Module):
    """An input convex network (Amos et al. 2017; Nystrom-Vestberg 2023, eq. (3.7)).

    ``z_1 = s(L_0 x + b_0)``, ``z_{j+1} = s(W_j z_j + L_j x + b_j)`` for the
    later hidden layers, and the output ``W_k z_k + L_k x + b_k``, with
    ``s`` the softplus, plus ``alpha |x|^2 / 2`` when ``alpha > 0``. Each
    ``W_j`` is the elementwise square of the trained parameter ``V_j``, so
    it is non-negative whatever ``V_j`` is; the ``L_j`` and ``b_j`` are
    free. Every parameter starts from the normal distribution with mean 0
    and standard deviation ``std``.

    Parameters
    ----------
    dim
        Input dimension.
    width, depth
        Units per hidden layer and number of hidden layers.
    std
        Standard deviation of the initial parameters.
    alpha
        Coefficient of the added ``|x|^2 / 2``, at least 0; with 0 nothing
        is added.
    """

    def __init__(
        self, dim: int, width: int, depth: int, std: float, alpha: float = 0.0
    ) -> None:
        super().__init__()
        self.inputs = torch.nn.ModuleList(
            [torch.nn.Linear(dim, width) for _ in range(depth)]
            + [torch.nn.Linear(dim, 1)]
        )
        self.roots = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(width, width)) for _ in range(depth - 1)]
            + [torch.nn.Parameter(torch.empty(1, width))]
        )
        self.alpha = alpha
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
        u = z.squeeze(-1)
        if self.alpha > 0.0:
            u = u + 0.5 * self.alpha * x.pow(2).sum(dim=-1)
        return u


class QuadraticForm(torch.nn.Module):
    """The QS form ``(x - x0)^T A^T A (x - x0) + b . x + c`` with ``A = alpha (I + B)``.

    ``B = smooth_scaling(Z, ratio)`` for the output ``Z`` of a plain
    network, reshaped to ``d x d``, so ``|B|_op < ratio < 1``.
    ``alpha = alpha_start exp(s)`` with ``s`` a trained scalar that starts
    at 0, so ``alpha`` is positive for every value of the parameters and
    ``A`` is invertible at every point, with singular values in
    ``alpha (1 - ratio, 1 + ratio)``. ``b`` and ``c`` are trained
    constants that start at 0. The output layer of the plain network starts
    at 0, so the training starts from ``A = alpha_start I`` everywhere,
    ``u = alpha_start^2 |x - x0|^2``.

    Parameters
    ----------
    dim
        Input dimension ``d``.
    width, depth, activation
        The plain network, as :func:`mlp` takes them, with ``d^2`` outputs.
    ratio
        ``epsilon / alpha``, in ``(0, 1)``.
    reference
        The reference point ``x0``, ``d`` numbers.
    alpha_start
        The start of ``alpha``, positive.
    report
        Plain values the solver records as ``Result.info["network"]``;
        :attr:`report` adds the current ``alpha`` to them.
    """

    reference: torch.Tensor

    def __init__(
        self,
        dim: int,
        width: int,
        depth: int,
        activation: str,
        ratio: float,
        reference: Sequence[float],
        alpha_start: float,
        report: dict,
    ) -> None:
        super().__init__()
        self.inner = mlp(dim, width, depth, activation, outputs=dim * dim)
        output = self.inner[-2]
        assert isinstance(output, torch.nn.Linear)
        with torch.no_grad():
            output.weight.zero_()
            output.bias.zero_()
        self.log_scale = torch.nn.Parameter(torch.zeros(()))
        self.b = torch.nn.Parameter(torch.zeros(dim))
        self.c = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("reference", torch.tensor([float(v) for v in reference]))
        self.dim, self.ratio, self.alpha_start = dim, ratio, alpha_start
        self.report = report

    @property
    def alpha(self) -> torch.Tensor:
        """``alpha = alpha_start exp(s)``, a positive scalar tensor."""
        return self.alpha_start * torch.exp(self.log_scale)

    @property
    def report(self) -> dict:
        """The report given, with ``alpha``, its current value, added."""
        return {**self._report, "alpha": float(self.alpha.detach())}

    @report.setter
    def report(self, value: dict) -> None:
        self._report = {key: v for key, v in value.items() if key != "alpha"}

    def matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``A`` at points of shape ``(M, d)``: shape ``(M, d, d)``."""
        z = self.inner(x).reshape(-1, self.dim, self.dim)
        eye = torch.eye(self.dim, dtype=x.dtype, device=x.device)
        return self.alpha * (eye + functional.smooth_scaling(z, self.ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.quadratic_form(
            self.matrix(x), x, self.reference, self.b, self.c
        )


class Switching(torch.optim.Optimizer):
    """Delegate each ``step`` to one torch optimizer for a number of calls, then to another.

    Both optimizers are built over the same parameters before the first
    step; each keeps its own state, so the second starts fresh when it takes
    over.

    Parameters
    ----------
    params
        The parameters, in one group.
    first, then
        The torch optimizers over ``params``.
    after
        The number of calls of :meth:`step` that go to ``first``.
    """

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        first: torch.optim.Optimizer,
        then: torch.optim.Optimizer,
        after: int,
    ) -> None:
        super().__init__(params, {})
        self.first, self.then, self.after = first, then, after
        self.calls = 0

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        """Take one step of ``first`` for the first ``after`` calls, of ``then`` after.

        Parameters
        ----------
        closure
            Zeroes the gradients, evaluates the loss, back-propagates it and
            returns it.

        Returns
        -------
        torch.Tensor
            What the optimizer's step returns.
        """
        optimizer = self.first if self.calls < self.after else self.then
        self.calls += 1
        return optimizer.step(closure)  # pyright: ignore[reportCallIssue, reportArgumentType]


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


def flat_grad(params: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate the gradients of ``params`` into one vector.

    A parameter without a gradient (one the loss does not reach, such as the
    output bias of a network whose Hessian is all the loss sees) contributes
    zeros.

    Parameters
    ----------
    params
        The parameters, in the order of the vector.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)`` for ``n`` parameter entries.
    """
    return torch.cat(
        [
            torch.zeros_like(p).view(-1) if p.grad is None else p.grad.view(-1)
            for p in params
        ]
    )


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
