"""Optimizers: the part of a neural solver that updates the network's parameters.

An optimizer part (ADR 0004) is a frozen dataclass of hyperparameters whose
:meth:`Optimizer.build` makes the torch optimizer. The training loop calls
the torch optimizer's ``step(closure)`` once per training step, full batch;
the closure evaluates the loss and its gradient, and its first call in a step
returns the evaluation the loop has already made at the current parameters.

This module imports torch only inside ``build``, so the parts can be
configured, compared and recorded without the ``neural`` extra.
"""

# No `from __future__ import annotations` in this module: jsonargparse reads the
# field types of the parts (ADR 0005); see maesolvers.classical.iterations.

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from maesolvers.core._params import integer, positive

if TYPE_CHECKING:
    import torch


class Optimizer(ABC):
    """Base class of every optimizer part (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements :meth:`build`.
    """

    @abstractmethod
    def build(
        self, parameters: "Iterable[torch.nn.Parameter]"
    ) -> "torch.optim.Optimizer":
        """Make the torch optimizer of the trainable parameters.

        Parameters
        ----------
        parameters
            The parameters to train.

        Returns
        -------
        torch.optim.Optimizer
            One call of its ``step(closure)`` is one training step.
        """


@dataclass(frozen=True, kw_only=True)
class LBFGS(Optimizer):
    """Limited-memory BFGS with a strong Wolfe line search, one iteration per step.

    ``torch.optim.LBFGS`` with ``max_iter=1``, so a training step is one
    quasi-Newton iteration and its line search, as an iteration of SciPy's
    BFGS is, and the curvature pairs carry over from step to step. Its own
    stopping tests are off; the solver stops on its ``tol``.

    Parameters
    ----------
    lr
        Scale of the initial step of the line search, positive (1 for a
        quasi-Newton method).
    history_size
        Curvature pairs kept, at least 1. Nystrom-Vestberg 2023 use the full
        BFGS of SciPy, which keeps a dense inverse Hessian approximation
        instead.

    Raises
    ------
    TypeError, ValueError
        On construction, for a value of the wrong type or out of range.
    """

    lr: float = 1.0
    history_size: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "lr", positive("lr", self.lr))
        object.__setattr__(
            self, "history_size", integer("history_size", self.history_size, 1)
        )

    def build(
        self, parameters: "Iterable[torch.nn.Parameter]"
    ) -> "torch.optim.Optimizer":
        """Make ``torch.optim.LBFGS`` with one iteration per step.

        Parameters
        ----------
        parameters
            The parameters to train.

        Returns
        -------
        torch.optim.LBFGS
        """
        import torch

        return torch.optim.LBFGS(
            parameters,
            lr=self.lr,
            max_iter=1,
            history_size=self.history_size,
            tolerance_grad=0.0,
            tolerance_change=0.0,
            line_search_fn="strong_wolfe",
        )


@dataclass(frozen=True, kw_only=True)
class BFGS(Optimizer):
    """BFGS with a dense inverse Hessian approximation, one iteration per step.

    The method of SciPy's ``minimize(method="BFGS")``, with which
    Nystrom-Vestberg 2023 trained: the approximation starts as the identity
    and gets the BFGS update after every iteration, and the step comes from
    a strong Wolfe line search (the one :class:`LBFGS` uses) started where
    SciPy starts its own, at the step that would repeat the last decrease
    of the loss. Unlike :class:`LBFGS` it keeps every curvature pair, in
    ``n^2`` numbers for ``n`` parameters, so it suits small networks; on
    the input convex network of the paper (563 parameters) it trains much
    further in the same number of steps (``benchmarks/icnn.md``, Finding 1).
    It has no parameters, and no stopping test of its own: the solver stops
    on its ``tol``.
    """

    def build(
        self, parameters: "Iterable[torch.nn.Parameter]"
    ) -> "torch.optim.Optimizer":
        """Make the dense BFGS optimizer.

        Parameters
        ----------
        parameters
            The parameters to train.

        Returns
        -------
        torch.optim.Optimizer
        """
        from maesolvers.neural import _torch

        return _torch.DenseBFGS(parameters)


@dataclass(frozen=True, kw_only=True)
class Adam(Optimizer):
    """Adam with a constant learning rate, one update per step.

    Parameters
    ----------
    lr
        Learning rate, positive.

    Raises
    ------
    TypeError, ValueError
        On construction, for a learning rate that is not a positive real.
    """

    lr: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(self, "lr", positive("lr", self.lr))

    def build(
        self, parameters: "Iterable[torch.nn.Parameter]"
    ) -> "torch.optim.Optimizer":
        """Make ``torch.optim.Adam``.

        Parameters
        ----------
        parameters
            The parameters to train.

        Returns
        -------
        torch.optim.Adam
        """
        import torch

        return torch.optim.Adam(parameters, lr=self.lr)
