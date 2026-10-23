"""NeuralSolver: a network, a loss, a sampler and an optimizer on one training loop.

The neural half of ADR 0003 and 0004: a :class:`NeuralSolver` is composed of
four parts and runs the shared training protocol of ADR 0008. Its ``setup``
draws the collocation points from the seed and builds the network, its
``run`` hands both to a Lightning ``Trainer`` in float64, and its
``finalize`` returns the trained network as ``Result.u`` with its Hessian by
automatic differentiation as ``Result.hessian``, both on numpy arrays, so
:func:`~maesolvers.metrics.evaluate` measures it on the shared evaluation
grid like any other result.

The solver and its parts need numpy only until a solve starts; torch and
lightning (the ``neural`` extra) are imported then.
"""

# No `from __future__ import annotations` here: jsonargparse reads the field
# types (ADR 0005); see maesolvers.classical.iterations.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from maesolvers.core._params import integer, positive
from maesolvers.core.boundary import Dirichlet, Transport
from maesolvers.core.callbacks import Callback
from maesolvers.core.domains import Domain
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solution, Solver, Trace, check_part
from maesolvers.neural.losses import Batch, Loss
from maesolvers.neural.networks import Network
from maesolvers.neural.optimizers import Optimizer
from maesolvers.neural.samplers import Collocation, Sampler

Accelerator = Literal["auto", "cpu", "gpu"]
"""Where to train: Lightning's ``accelerator``, on one device."""

_HINT = (
    "neural solvers need torch and lightning, in the neural extra: pip install "
    '"maesolvers[neural] @ git+https://github.com/jczhang02/maesolvers"'
)


def _require_extra() -> None:
    """Raise an ImportError that names the extra if torch or lightning is missing."""
    try:
        import lightning  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(_HINT) from error


@dataclass(frozen=True, eq=False)
class _Context:
    """The working data of one solve: the points, the network, the data there."""

    collocation: Collocation
    model: Any
    f: np.ndarray
    g: np.ndarray | None
    f_boundary: np.ndarray
    target: Domain | None


@dataclass(frozen=True, eq=False)
class _Trained:
    """The state :meth:`NeuralSolver.run` returns: the network and how training ended."""

    model: Any
    stop: str
    device: str
    threads: int


@dataclass(frozen=True, kw_only=True)
class NeuralSolver(Solver[_Context, _Trained]):
    """A neural solver: a network trained on a loss at collocation points (ADR 0004, 0008).

    The plug-in point for a neural method: pass your own part to any of the
    four fields. The training is full batch, in float64, one optimizer step
    per training step. A step evaluates the loss and its gradient at the
    current parameters and stops the training, without an update, when the
    largest gradient component is at most ``tol`` (converged) or the loss
    is not finite (diverged); otherwise it takes the optimizer step, and the
    training stops unconverged when the step leaves the parameters
    unchanged or after ``max_steps`` steps. The solver supports a problem
    when its network, loss and sampler do (issue #23; an optimizer does not
    depend on the problem); a preset also declares its support on the
    class.

    Parameters
    ----------
    network
        The network part (:class:`~maesolvers.neural.networks.Network`).
    loss
        The loss part (:class:`~maesolvers.neural.losses.Loss`).
    sampler
        The sampler part (:class:`~maesolvers.neural.samplers.Sampler`).
    optimizer
        The optimizer part (:class:`~maesolvers.neural.optimizers.Optimizer`).
    max_steps
        Most training steps, at least 1.
    tol
        Stop, converged, when the largest component of the loss gradient is
        at most this; positive.
    seed
        Seeds the collocation points (a numpy generator) and the initial
        network (torch's CPU generator, restored after the build); at least
        0. Seeds 0 to 9 are the seeds of a reported run (ADR 0008).
    accelerator
        ``"cpu"`` (the default, so that the seed fixes the result on any
        machine with the same thread count), ``"gpu"`` or ``"auto"`` (a GPU
        when there is one); the training uses one device. Devices round
        differently, so a run on another device drifts from a CPU run.

    Raises
    ------
    TypeError
        On construction, if a part is a class, not an instance of its base,
        or of a class without its own frozen dataclass decorator and
        keyword-only fields; for a parameter of the wrong type.
    ValueError
        On construction, for a parameter out of range.
    ImportError
        From :meth:`solve`, without torch or lightning.

    Notes
    -----
    ``Result.info`` holds, besides the fields of every solver, ``stop`` (why
    the training stopped, as :class:`~maesolvers.core.result.Result` defines
    it: ``"converged"``, ``"stalled"``, ``"diverged"`` or
    ``"max_steps"``), ``loss_terms`` (the loss terms of the trained network,
    ``"loss"`` among them), ``collocation`` (``interior_points``,
    ``boundary_points`` and :meth:`Collocation.fingerprint
    <maesolvers.neural.samplers.Collocation.fingerprint>`), ``device``,
    ``threads`` (torch's CPU threads, on which the rounding, and so the
    result, depends) and, when the network's module carries a ``report``
    (see :meth:`Network.build <maesolvers.neural.networks.Network.build>`),
    ``network``, that report.
    ``iterations`` counts the steps that updated the parameters, ``history``
    holds the loss at the start of each of them and ``unknowns`` the trained
    parameters. Each step's metrics, passed to ``on_step_end``, are the loss
    terms by name.
    """

    network: Network
    loss: Loss
    sampler: Sampler
    optimizer: Optimizer
    max_steps: int = 1000
    tol: float = 1e-5
    seed: int = 0
    accelerator: Accelerator = "cpu"

    def __post_init__(self) -> None:
        super().__post_init__()
        name = type(self).__name__
        check_part(f"{name}.network", self.network, Network)
        check_part(f"{name}.loss", self.loss, Loss)
        check_part(f"{name}.sampler", self.sampler, Sampler)
        check_part(f"{name}.optimizer", self.optimizer, Optimizer)
        object.__setattr__(self, "max_steps", integer("max_steps", self.max_steps, 1))
        object.__setattr__(self, "tol", positive("tol", self.tol))
        object.__setattr__(self, "seed", integer("seed", self.seed, 0))
        if self.accelerator not in ("auto", "cpu", "gpu"):
            raise ValueError(
                f"accelerator must be 'auto', 'cpu' or 'gpu', got {self.accelerator!r}"
            )

    def supports(self, problem: Problem) -> bool:
        """Whether the network, the loss and the sampler all support the problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return (
            self.network.supports(problem)
            and self.loss.supports(problem)
            and self.sampler.supports(problem)
        )

    def check(self, problem: Problem) -> None:
        """Raise if a part does not support the problem or the neural extra is missing.

        Parameters
        ----------
        problem
            The problem to solve.

        Raises
        ------
        UnsupportedProblemError
            If a part does not support the problem.
        ImportError
            Without torch or lightning; the message names the extra.
        """
        super().check(problem)
        _require_extra()

    def collocation(self, problem: Problem) -> Collocation:
        """Draw the collocation points of a problem with the sampler, from the seed.

        The same for every call, and for every solver with an equal sampler
        and seed.

        Parameters
        ----------
        problem
            A problem the sampler supports.

        Returns
        -------
        Collocation
        """
        return self.sampler.sample(problem, np.random.default_rng(self.seed))

    def setup(self, problem: Problem) -> _Context:
        """Draw the collocation points and build the network from the seed.

        Parameters
        ----------
        problem
            A problem that passed :meth:`check`.

        Returns
        -------
        _Context
            The points, the network, ``f`` at the interior points, ``g``
            (None for a boundary condition other than Dirichlet) and ``f``
            at the boundary points, and the transport target (None for a
            boundary condition other than Transport).
        """
        from maesolvers.neural import _torch

        collocation = self.collocation(problem)
        with _torch.building(self.seed):
            model = self.network.build(problem, collocation)
        boundary, points = problem.boundary, collocation.boundary
        g = None
        if isinstance(boundary, Dirichlet):
            g = boundary.g(points) if len(points) else np.zeros(0)
        f_boundary = problem.f(points) if len(points) else np.zeros(0)
        target = boundary.target if isinstance(boundary, Transport) else None
        return _Context(
            collocation, model, problem.f(collocation.interior), g, f_boundary, target
        )

    def run(
        self, problem: Problem, context: _Context, callbacks: Sequence[Callback]
    ) -> Trace[_Trained]:
        """Train the network on a Lightning Trainer.

        Parameters
        ----------
        problem
            The problem.
        context
            What :meth:`setup` returned.
        callbacks
            Each gets ``on_step_end`` with the loss terms after every step.

        Returns
        -------
        Trace
        """
        import torch

        from maesolvers.neural import _training

        threads = torch.get_num_threads()
        module, steps, device = _training.train(
            self, context.model, self._batch(context), callbacks
        )
        state = _Trained(module.model.cpu(), module.stop, device, threads)
        return Trace(state, steps, module.stop == "converged", tuple(module.history))

    def finalize(
        self, problem: Problem, context: _Context, state: _Trained
    ) -> Solution:
        """Return the network as ``u`` and its autograd Hessian as ``hessian``.

        Parameters
        ----------
        problem
            The problem.
        context
            What :meth:`setup` returned.
        state
            The trained network and how the training ended.

        Returns
        -------
        Solution
            ``u`` and ``hessian`` take points of shape ``(..., d)`` and
            return float64 arrays of shape ``(...)`` and ``(..., d, d)``.
        """
        from maesolvers.neural import _torch

        model, dim = state.model.eval(), problem.dim
        terms = self.loss(model, self._batch(context))
        collocation = context.collocation
        info: dict[str, Any] = {
            "stop": state.stop,
            "loss_terms": {name: float(value) for name, value in terms.items()},
            "collocation": {
                "interior_points": len(collocation.interior),
                "boundary_points": len(collocation.boundary),
                "fingerprint": collocation.fingerprint(),
            },
            "device": state.device,
            "threads": state.threads,
        }
        report = getattr(model, "report", None)
        if report is not None:
            info["network"] = dict(report)

        def u(points: np.ndarray) -> np.ndarray:
            return _torch.values(model, points, dim)

        def hessian(points: np.ndarray) -> np.ndarray:
            return _torch.hessians(model, points, dim)

        return Solution(
            u=u,
            hessian=hessian,
            unknowns=sum(p.numel() for p in _torch.trainable(model)),
            info=info,
        )

    @staticmethod
    def _batch(context: _Context) -> Batch:
        """Return the collocation points and data of a solve as float64 CPU tensors."""
        import torch

        def tensor(array: np.ndarray) -> "torch.Tensor":
            return torch.tensor(np.asarray(array, dtype=float), dtype=torch.float64)

        points = context.collocation
        return Batch(
            interior=tensor(points.interior),
            f=tensor(context.f),
            boundary=tensor(points.boundary),
            g=None if context.g is None else tensor(context.g),
            f_boundary=tensor(context.f_boundary),
            target=context.target,
        )
