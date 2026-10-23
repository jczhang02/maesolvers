"""Networks: the part of a neural solver that represents the solution.

A network part (ADR 0004) is a frozen dataclass of hyperparameters whose
:meth:`Network.build` makes the torch module ``u`` for one problem. The
module maps points of shape ``(M, d)`` to values of shape ``(M,)``. The
solver calls ``build`` once per solve, with float64 as torch's default
dtype and torch's random generator seeded from the solver's seed, on the
CPU; the training loop moves the module to its device.

This module imports torch only inside ``build``, so the parts can be
configured, compared and recorded without the ``neural`` extra.
"""

# No `from __future__ import annotations` in this module: jsonargparse reads the
# field types of the parts (ADR 0005); see maesolvers.classical.iterations.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from maesolvers.core._params import integer, nonnegative, positive
from maesolvers.core.boundary import Dirichlet
from maesolvers.core.domains import BoundaryDistance
from maesolvers.core.problem import Problem
from maesolvers.core.solver import check_part
from maesolvers.neural.samplers import Collocation

if TYPE_CHECKING:
    import torch

Activation = Literal["sigmoid", "tanh", "softplus"]
"""The smooth activations :class:`MLP` offers; ReLU has no second derivative."""


class Network(ABC):
    """Base class of every network part (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements :meth:`build`,
    and overrides :meth:`supports` if it does not handle every problem.
    """

    def supports(self, problem: Problem) -> bool:
        """Whether the network can represent the problem's solution; True by default.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return True

    @abstractmethod
    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Make the module ``u`` for one problem.

        Parameters
        ----------
        problem
            A problem the network supports.
        collocation
            The collocation points of the solve, for a network that fits
            something to the data before training.

        Returns
        -------
        torch.nn.Module
            Maps points of shape ``(M, d)`` to values of shape ``(M,)``;
            the parameters that require a gradient are the ones the solver
            trains. A module may carry an attribute ``report``, a mapping of
            plain values that says how a fit inside ``build`` went; the
            solver records it as ``Result.info["network"]``.
        """


@dataclass(frozen=True, kw_only=True)
class MLP(Network):
    """A fully connected feed-forward network with a linear output layer.

    ``depth`` hidden layers of ``width`` units each, with the same smooth
    activation, then one linear output unit, so ``width = 10, depth = 5`` in
    2D has 481 parameters (Nystrom-Vestberg 2023, Example 4.1). Every weight
    and bias starts from the standard normal distribution, as in that paper
    (Section 4).

    Parameters
    ----------
    width
        Units per hidden layer, at least 1.
    depth
        Number of hidden layers, at least 1.
    activation
        ``"sigmoid"``, ``"tanh"`` or ``"softplus"``.

    Raises
    ------
    TypeError, ValueError
        On construction, for a size that is not an integer or is less than 1,
        or an unknown activation.
    """

    width: int = 10
    depth: int = 5
    activation: Activation = "sigmoid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", integer("width", self.width, 1))
        object.__setattr__(self, "depth", integer("depth", self.depth, 1))
        if self.activation not in ("sigmoid", "tanh", "softplus"):
            raise ValueError(
                "activation must be 'sigmoid', 'tanh' or 'softplus', "
                f"got {self.activation!r}"
            )

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Make the network for the problem's dimension.

        Parameters
        ----------
        problem
            Any problem.
        collocation
            Not used.

        Returns
        -------
        torch.nn.Module
        """
        from maesolvers.neural import _torch

        return _torch.mlp(problem.dim, self.width, self.depth, self.activation)


@dataclass(frozen=True, kw_only=True)
class ICNN(Network):
    """An input convex neural network: convex in its input by construction.

    The construction of Amos, Xu and Kolter (2017), in the form of
    Nystrom-Vestberg 2023, eq. (3.7)::

        z_1     = s(L_0 x + b_0)
        z_{j+1} = s(W_j z_j + L_j x + b_j),   j = 1, ..., depth - 1
        u(x)    = W_depth z_depth + L_depth x + b_depth

    with ``s`` the softplus, ``log(1 + e^t)``. Each ``z_j`` is convex in
    ``x``: an affine map of ``x`` is convex, a non-negative combination of
    convex functions is convex, and a convex non-decreasing function of a
    convex function is convex. So every ``W_j`` must be non-negative and the
    activation convex and non-decreasing; the ``L_j`` and ``b_j`` are free.
    As in the paper (Section 4), ``W_j`` is the elementwise square
    ``V_j^2`` of the trained parameter ``V_j``, and every parameter starts
    from a normal distribution with mean 0. The output layer is linear, so
    ``u`` can take any value. ``width = 10, depth = 5`` in 2D has 563
    parameters (Example 4.2).

    With ``alpha > 0`` the network is ``u + alpha |x|^2 / 2``, whose
    Hessian is at least ``alpha I``: uniformly convex, with no new
    parameter. The QS-PINN note (Section 2) takes it as a baseline, since
    the plain network only has a positive semidefinite Hessian; ``alpha``
    must not exceed the smallest Hessian eigenvalue of the solution, or the
    network cannot represent it.

    The softplus is the one activation offered: it is convex, increasing and
    smooth, so ``u`` has a Hessian everywhere; ReLU is convex but has no
    second derivative, and tanh and the sigmoid are not convex.

    Parameters
    ----------
    width
        Units per hidden layer, at least 1.
    depth
        Number of hidden layers, at least 1.
    init_std
        Standard deviation of every initial parameter, the roots ``V_j``
        included; positive. The paper says "the normal distribution" and
        gives no standard deviation.
    alpha
        Coefficient of the added ``|x|^2 / 2``, at least 0; 0 (the default)
        adds nothing.

    Raises
    ------
    TypeError, ValueError
        On construction, for a size that is not an integer or is less than
        1, an ``init_std`` that is not a positive real, or an ``alpha`` that
        is negative or not a finite real.
    """

    width: int = 10
    depth: int = 5
    init_std: float = 1.0
    alpha: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", integer("width", self.width, 1))
        object.__setattr__(self, "depth", integer("depth", self.depth, 1))
        object.__setattr__(self, "init_std", positive("init_std", self.init_std))
        object.__setattr__(self, "alpha", nonnegative("alpha", self.alpha))

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Make the network for the problem's dimension.

        Parameters
        ----------
        problem
            Any problem.
        collocation
            Not used.

        Returns
        -------
        torch.nn.Module
            Its trainable parameters are the ``V_j``, ``L_j`` and ``b_j``.
        """
        from maesolvers.neural import _torch

        return _torch.InputConvex(
            problem.dim, self.width, self.depth, self.init_std, self.alpha
        )


@dataclass(frozen=True, kw_only=True)
class HardDirichlet(Network):
    """The ansatz ``u = G + D N``, which meets the Dirichlet condition by construction.

    Nystrom-Vestberg 2023, eq. (3.1), after Berg and Nystrom 2018: ``N`` is
    the trained network; ``G`` is a network fitted to the boundary data
    ``g`` at the collocation boundary points; ``D`` is a network fitted to
    the distance to the boundary, at the first ``distance_points`` interior
    collocation points and at the boundary points (where it is 0). ``G`` and
    ``D`` have one hidden layer of tanh units, are fitted by L-BFGS on the
    mean squared error before training, and stay fixed during it. On the
    boundary ``u - g = (G - g) + D N``: the misfit of ``G`` plus what ``N``
    leaks through the residue of ``D``, which grows with ``N``.

    The module's ``report`` (``Result.info["network"]``) holds, for each fit
    under ``"lift"`` and ``"distance"``, what :func:`fit` reports
    (``iterations``, ``max_gradient``, ``converged``, ``mse``,
    ``max_misfit``), and ``lift_boundary_misfit`` (largest ``|G - g|``) and
    ``distance_boundary_max`` (largest ``|D|``) at the boundary collocation
    points.

    Parameters
    ----------
    network
        The network part ``N``.
    lift_width
        Hidden units of ``G``, at least 1 (20 in the paper).
    distance_width
        Hidden units of ``D``, at least 1 (40 in the paper).
    distance_points
        Interior points at which ``D`` is fitted, at least 1 (300 in the
        paper); at most the number of interior collocation points are used.
    fit_steps
        Most L-BFGS iterations of each fit, at least 1. The paper gives no
        cap (SciPy's BFGS, whose own default cap is 200 iterations per
        parameter); this one is large enough for both fits to meet
        ``fit_tol`` on T1 and T6 at the paper's settings (at most 557
        iterations for ``G`` and 2752 for ``D``), and the report says
        whether a fit hit it.
    fit_tol
        Each fit stops when the largest component of its gradient is at most
        this, positive (1e-5, the default ``gtol`` of the SciPy BFGS the
        paper used).

    Raises
    ------
    TypeError
        On construction, if ``network`` is not an instance of a well-defined
        :class:`Network`.
    TypeError, ValueError
        On construction, for a count that is not an integer or is less than
        1, or a ``fit_tol`` that is not a positive real.
    """

    network: Network
    lift_width: int = 20
    distance_width: int = 40
    distance_points: int = 300
    fit_steps: int = 10_000
    fit_tol: float = 1e-5

    def __post_init__(self) -> None:
        check_part(f"{type(self).__name__}.network", self.network, Network)
        for name in ("lift_width", "distance_width", "distance_points", "fit_steps"):
            object.__setattr__(self, name, integer(name, getattr(self, name), 1))
        object.__setattr__(self, "fit_tol", positive("fit_tol", self.fit_tol))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is Dirichlet on a domain that knows its boundary distance.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return (
            isinstance(problem.boundary, Dirichlet)
            and isinstance(problem.domain, BoundaryDistance)
            and self.network.supports(problem)
        )

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Fit ``G`` and ``D``, then return ``G + D N`` with ``G`` and ``D`` frozen.

        Parameters
        ----------
        problem
            A problem the ansatz supports.
        collocation
            The collocation points; ``G`` and ``D`` are fitted at them.

        Returns
        -------
        torch.nn.Module
            Its trainable parameters are those of ``N``; its ``report`` says
            how the fits went.

        Raises
        ------
        ValueError
            If the collocation set has no boundary points.
        """
        from maesolvers.neural import _torch

        if len(collocation.boundary) == 0:
            raise ValueError(
                "HardDirichlet fits G and D at the boundary points; the sampler "
                "gave none"
            )
        assert isinstance(problem.boundary, Dirichlet)  # supports checked it
        assert isinstance(problem.domain, BoundaryDistance)
        boundary = np.asarray(collocation.boundary)
        inner = np.asarray(collocation.interior[: self.distance_points])
        g = problem.boundary.g(boundary)
        dim = problem.dim
        lift = _torch.mlp(dim, self.lift_width, 1, "tanh")
        lift_fit = _torch.fit(lift, boundary, g, self.fit_steps, self.fit_tol)
        distance = _torch.mlp(dim, self.distance_width, 1, "tanh")
        distance_fit = _torch.fit(
            distance,
            np.concatenate([inner, boundary]),
            np.concatenate(
                [problem.domain.boundary_distance(inner), np.zeros(len(boundary))]
            ),
            self.fit_steps,
            self.fit_tol,
        )
        report = {
            "lift": lift_fit,
            "distance": distance_fit,
            "lift_boundary_misfit": float(
                np.max(np.abs(_torch.values(lift, boundary, dim) - g))
            ),
            "distance_boundary_max": float(
                np.max(np.abs(_torch.values(distance, boundary, dim)))
            ),
        }
        network = self.network.build(problem, collocation)
        return _torch.Ansatz(lift, distance, network, report)


@dataclass(frozen=True, kw_only=True)
class IdentityStart(Network):
    """A network whose training starts from ``grad u = x``: fitted to the identity map first.

    Caboussat-Peruso 2025 (Section 4.1, after Singh, Bauer and Joshi 2022) start the
    training of their input convex network from parameters fitted so that
    ``grad u`` approximates the identity map, ``u`` about ``|x|^2 / 2``,
    which is convex and maps each point to itself; from the random start,
    an input convex network is hard to train. ``build`` builds the network
    part, fits the gradient of its module to ``x`` by L-BFGS on the mean
    squared misfit at the interior and boundary collocation points, and
    returns the module, all of whose parameters stay trainable. The paper
    gives neither the points, the optimizer nor the length of the fit.

    The module's ``report`` (``Result.info["network"]``) holds the network
    part's own report, if it has one, and under ``"identity_fit"`` what the
    fit reports (``iterations``, ``max_gradient``, ``converged``, ``mse``,
    ``max_misfit``, the largest misfit of one component of the gradient).

    Parameters
    ----------
    network
        The network part to fit, such as
        :class:`~maesolvers.neural.networks.ICNN`.
    fit_steps
        Most L-BFGS iterations of the fit, at least 1.
    fit_tol
        The fit stops when the largest component of its gradient is at most
        this, positive.

    Raises
    ------
    TypeError
        On construction, if ``network`` is not an instance of a well-defined
        :class:`Network`.
    TypeError, ValueError
        On construction, for a ``fit_steps`` that is not an integer or is
        less than 1, or a ``fit_tol`` that is not a positive real.
    """

    network: Network
    fit_steps: int = 1000
    fit_tol: float = 1e-6

    def __post_init__(self) -> None:
        check_part(f"{type(self).__name__}.network", self.network, Network)
        object.__setattr__(self, "fit_steps", integer("fit_steps", self.fit_steps, 1))
        object.__setattr__(self, "fit_tol", positive("fit_tol", self.fit_tol))

    def supports(self, problem: Problem) -> bool:
        """Whether the network part supports the problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return self.network.supports(problem)

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Build the network part's module and fit its gradient to the identity map.

        Parameters
        ----------
        problem
            A problem the network supports.
        collocation
            The collocation points; the gradient is fitted at the interior and
            boundary points.

        Returns
        -------
        torch.nn.Module
            The network part's module, fitted, with its ``report``.
        """
        from maesolvers.neural import _torch

        module = self.network.build(problem, collocation)
        points = np.concatenate([collocation.interior, collocation.boundary])
        fit = _torch.fit_gradient(module, points, points, self.fit_steps, self.fit_tol)
        report = {**getattr(module, "report", {}), "identity_fit": fit}
        module.report = report  # pyright: ignore[reportArgumentType]
        return module


@dataclass(frozen=True, kw_only=True)
class QuadraticForm(Network):
    """The QS form of QS-PINN: ``u = (x - x0)^T A(x)^T A(x) (x - x0) + b . x + c``.

    JC Zhang's QS-PINN note (``writing/notes/qs-pinn/main.tex`` in
    jczhang02/monge-ampere-equations), Definition 3.1 with the offset of
    Section 6, eq. (9) and the smooth scaling of eq. (11)::

        A(x) = alpha (I + B(x)),   B = ratio Z / sqrt(1 + |Z|_F^2)

    with ``Z(x)`` the ``d x d`` output of a plain network (``width``,
    ``depth``, ``activation``, standard normal start as :class:`MLP`, but
    the output layer at 0), and ``alpha``, ``b`` and ``c`` trained
    constants. Whatever the parameters, ``u`` lies above its tangent plane
    ``c + b . x`` at the reference point ``x0``, where its Hessian is
    ``2 A^T A`` (Proposition 3.3). ``|B|_op <= |B|_F < ratio < 1``, so
    ``A`` is invertible at every point: the bound ``epsilon`` of eq. (9) is
    ``ratio |alpha|``, below ``|alpha|`` for every trained ``alpha``, and
    the Hessian at ``x0`` is positive definite, the condition on the
    structure of the residual certificate (Corollary 5.1). ``ratio`` trades
    the spread of the singular values of ``A`` at a point, at most
    ``(1 + ratio) / (1 - ratio)``, against the margin
    ``sigma_min(A) >= |alpha| (1 - ratio)`` (Proposition 4.4 (a)).

    ``alpha`` starts at ``f(x0)^(1 / (2 d)) / sqrt(2)`` (Section 6), and
    ``b``, ``c`` and ``B`` at 0, so the training starts from
    ``u = |x - x0|^2 f(x0)^(1/d) / 2``, whose Hessian determinant is
    ``f(x0)`` and whose gradient map is ``f(x0)^(1/d) (x - x0)``, the
    identity map when ``f(x0) = 1``. The module's ``report``
    (``Result.info["network"]``) holds ``reference`` and ``alpha_start``.

    The network supports a problem when ``x0`` lies inside its domain and
    ``f(x0)`` is positive and finite: the form gives
    ``det D^2 u(x0) = 2^d det(A^T A) > 0``, so it cannot meet
    ``f(x0) = 0`` (T20 at the centre of its box).

    Parameters
    ----------
    width, depth
        Units per hidden layer and hidden layers of the plain network, each
        at least 1.
    activation
        ``"tanh"`` (the note's choice), ``"sigmoid"`` or ``"softplus"``.
    ratio
        ``epsilon / alpha``, in ``(0, 1)``.
    reference
        The reference point ``x0``, ``d`` finite numbers; None (the default)
        for the centre of the domain's bounding box.

    Raises
    ------
    TypeError, ValueError
        On construction, for a size that is not an integer or is less than
        1, an unknown activation, a ``ratio`` outside ``(0, 1)``, or a
        ``reference`` that is not a sequence of finite numbers.
    """

    width: int = 10
    depth: int = 5
    activation: Activation = "tanh"
    ratio: float = 0.6
    reference: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", integer("width", self.width, 1))
        object.__setattr__(self, "depth", integer("depth", self.depth, 1))
        if self.activation not in ("sigmoid", "tanh", "softplus"):
            raise ValueError(
                "activation must be 'sigmoid', 'tanh' or 'softplus', "
                f"got {self.activation!r}"
            )
        ratio = positive("ratio", self.ratio)
        if not ratio < 1.0:
            raise ValueError(f"ratio must be less than 1, got {ratio}")
        object.__setattr__(self, "ratio", ratio)
        if self.reference is not None:
            object.__setattr__(self, "reference", _point("reference", self.reference))

    def point(self, problem: Problem) -> np.ndarray:
        """Return the reference point ``x0`` for the problem.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        numpy.ndarray
            ``reference``, or the centre of the domain's bounding box.
        """
        if self.reference is not None:
            return np.array(self.reference)
        box = problem.domain.bbox
        return (np.array(box.lo) + np.array(box.hi)) / 2.0

    def supports(self, problem: Problem) -> bool:
        """Whether ``x0`` lies inside the domain and ``f(x0)`` is positive and finite.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        x0 = self.point(problem)
        if len(x0) != problem.dim or not problem.domain.interior(x0[None])[0]:
            return False
        f0 = float(problem.f(x0[None])[0])
        return bool(np.isfinite(f0) and f0 > 0.0)

    def build(self, problem: Problem, collocation: Collocation) -> "torch.nn.Module":
        """Make the form for the problem, with ``alpha`` from ``f(x0)``.

        Parameters
        ----------
        problem
            A problem the network supports.
        collocation
            Not used.

        Returns
        -------
        torch.nn.Module
            Its trainable parameters are those of the plain network,
            ``alpha``, ``b`` and ``c``; its ``report`` holds ``reference``
            and ``alpha_start``.
        """
        from maesolvers.neural import _torch

        x0 = self.point(problem)
        dim = problem.dim
        alpha = float(problem.f(x0[None])[0]) ** (1.0 / (2 * dim)) / math.sqrt(2.0)
        reference = [float(v) for v in x0]
        report = {"reference": reference, "alpha_start": alpha}
        return _torch.QuadraticStructure(
            dim,
            self.width,
            self.depth,
            self.activation,
            self.ratio,
            reference,
            alpha,
            report,
        )


def _point(name: str, value: object) -> tuple[float, ...]:
    """Check a point given as a sequence of finite numbers; return it as floats."""
    try:
        point = tuple(float(v) for v in value)  # type: ignore[union-attr]
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of numbers, got {value!r}"
        ) from None
    if len(point) == 0 or not all(np.isfinite(point)):
        raise ValueError(f"{name} must be one or more finite numbers, got {point}")
    return point
