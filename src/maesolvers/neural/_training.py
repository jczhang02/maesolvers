"""The shared training loop of the neural solvers: a LightningModule on a Trainer (ADR 0008).

Imported only when a neural solver runs, so that ``import maesolvers`` needs
neither torch nor lightning (ADR 0006).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import lightning
import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, IterableDataset

from maesolvers.core.callbacks import Callback
from maesolvers.neural import _torch
from maesolvers.neural.losses import Batch, Loss
from maesolvers.neural.optimizers import Optimizer

if TYPE_CHECKING:
    from maesolvers.neural.solver import NeuralSolver

_LOGGERS = ("lightning.pytorch", "lightning.fabric")
"""Lightning's loggers, which report the hardware at INFO level on every fit."""

_ADVICE = (
    "does not have many workers",
    "`srun` command is available",
    "GPU available but not used",
)
"""Lightning's warnings about the environment, silenced on every fit.

The last fires only for ``accelerator="cpu"`` on a host with a GPU: the CPU
is the solver's documented default (ADR 0008), so the warning is noise.
"""


class _Forever(IterableDataset):
    """An endless stream of empty batches: the data live on the module (full batch)."""

    def __iter__(self) -> Iterator[int]:
        while True:
            yield 0


class TrainingModule(lightning.LightningModule):
    """One full-batch optimizer step per training step, with the solver's stopping tests.

    A step evaluates the loss and its gradient at the current parameters.
    It stops the training, without an update, when the loss is not finite
    (``"diverged"``) or the largest gradient component is at most ``tol``
    (``"converged"``); otherwise it calls the optimizer's ``step(closure)``
    and stops when the update left the parameters unchanged
    (``"stalled"``), recording nothing for that step. The step's metrics are the terms of that first
    evaluation, the loss of the parameters the step starts from.

    Parameters
    ----------
    model
        The module ``u`` to train.
    loss, optimizer
        The solver's loss and optimizer parts.
    batch
        The collocation points and data; moved to the training device when
        the fit starts.
    tol
        Stop when the largest gradient component is at most this.

    Attributes
    ----------
    metrics : dict
        The terms of the last step that updated the parameters, as floats.
    history : list of float
        The loss of every step that updated the parameters.
    stop : str
        Why the training stopped: ``"converged"``, ``"stalled"``,
        ``"diverged"``, or ``"max_steps"`` if none of these.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Loss,
        optimizer: Optimizer,
        batch: Batch,
        tol: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.data = batch
        self._loss = loss
        self._optimizer = optimizer
        self._tol = tol
        self.automatic_optimization = False
        self.metrics: dict[str, float] = {}
        self.history: list[float] = []
        self.stop = "max_steps"
        self.stepped = False

    def on_fit_start(self) -> None:
        """Move the collocation points and data to the training device."""
        self.data = self.data.to(self.device)

    def _torch_optimizer(self) -> LightningOptimizer:
        optimizer = self.optimizers()
        assert isinstance(optimizer, LightningOptimizer)  # one optimizer
        return optimizer

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the optimizer part over the model's trainable parameters."""
        return self._optimizer.build(_torch.trainable(self.model))

    def _evaluate(self) -> dict[str, torch.Tensor]:
        """Loss terms at the current parameters, with their gradient in ``.grad``."""
        self._torch_optimizer().zero_grad()
        terms = self._loss(self.model, self.data)
        self.manual_backward(terms["loss"])
        return terms

    def _finish(self, reason: str) -> None:
        self.stop = reason
        self.trainer.should_stop = True

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluate, test the stopping criteria, then take one optimizer step."""
        self.stepped = False
        parameters = _torch.trainable(self.model)
        terms = self._evaluate()
        loss = terms["loss"]
        if not torch.isfinite(loss):
            self._finish("diverged")
            return
        gradient = _torch.flat_grad(parameters)
        if float(gradient.abs().max()) <= self._tol:
            self._finish("converged")
            return
        before = parameters_to_vector(parameters).detach().clone()
        first = [True]

        def closure() -> torch.Tensor:
            if first[0]:  # the evaluation above, at the same parameters
                first[0] = False
                return loss
            return self._evaluate()["loss"]

        self._torch_optimizer().step(closure=closure)
        if torch.equal(before, parameters_to_vector(parameters)):
            self._finish("stalled")  # not a step: nothing is recorded
            return
        self.metrics = {name: float(value) for name, value in terms.items()}
        self.history.append(self.metrics["loss"])
        self.stepped = True


class _Forward(lightning.Callback):
    """Forward the end of every step to the solve's callbacks (ADR 0003)."""

    def __init__(self, solver: NeuralSolver, callbacks: Sequence[Callback]) -> None:
        self._solver = solver
        self._callbacks = callbacks

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert isinstance(pl_module, TrainingModule)
        if pl_module.stepped:
            for callback in self._callbacks:
                callback.on_step_end(
                    self._solver, len(pl_module.history), dict(pl_module.metrics)
                )


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence Lightning's per-fit hardware report and its advice in ``_ADVICE``."""
    levels = {name: logging.getLogger(name).level for name in _LOGGERS}
    for name in _LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            for advice in _ADVICE:
                warnings.filterwarnings("ignore", message=f".*{advice}.*")
            yield
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def train(
    solver: NeuralSolver,
    model: torch.nn.Module,
    batch: Batch,
    callbacks: Sequence[Callback],
) -> tuple[TrainingModule, int, str]:
    """Train ``model`` with the solver's loss and optimizer on a Lightning Trainer.

    Parameters
    ----------
    solver
        The solver: its loss, optimizer, ``max_steps``, ``tol`` and
        ``accelerator``.
    model
        The module to train, in float64 on the CPU.
    batch
        The collocation points and data, on the CPU.
    callbacks
        Each gets ``on_step_end`` after every step that updated the
        parameters.

    Returns
    -------
    tuple
        The training module (its ``model`` trained, on the training device),
        the number of steps that updated the parameters, and the type of the
        device that trained (``"cpu"``, ``"cuda"``).
    """
    module = TrainingModule(model, solver.loss, solver.optimizer, batch, solver.tol)
    with _quiet():
        trainer = lightning.Trainer(
            accelerator=solver.accelerator,
            devices=1,
            precision="64-true",
            max_steps=solver.max_steps,
            max_epochs=-1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[_Forward(solver, callbacks)],
        )
        # Its own generator: a data loader draws a seed from torch's global one.
        loader = DataLoader(_Forever(), batch_size=None, generator=torch.Generator())
        trainer.fit(module, train_dataloaders=loader)
    return module, len(module.history), trainer.strategy.root_device.type
