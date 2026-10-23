"""Samplers: the part of a neural solver that picks the collocation points.

A sampler (ADR 0004) draws the collocation points of one problem from a
random generator: points inside the domain, where the loss evaluates the
equation, and points on the boundary, where it evaluates the boundary
condition. :class:`~maesolvers.neural.solver.NeuralSolver` seeds the
generator, so the points are drawn from the seed by the sampler. ADR 0008
asks for one stored collocation set per problem, shared by every method;
whether that set is per problem or per sampler, and whether it is persisted,
is open (issue #52). Everything here needs numpy only.
"""

# No `from __future__ import annotations` in this module: jsonargparse reads the
# field types of the parts (ADR 0005); see maesolvers.classical.iterations.

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from maesolvers.core._params import integer
from maesolvers.core.boundary import Transport
from maesolvers.core.domains import BoundarySampling, Domain
from maesolvers.core.problem import Problem

_REJECTION_ROUNDS = 1000
"""Rounds of rejection sampling before a domain counts as too thin to sample."""


@dataclass(frozen=True, eq=False)
class Collocation:
    """The collocation points of one problem.

    Stored as read-only float copies, so the points of a solve cannot change
    after the sampler drew them.

    Parameters
    ----------
    interior
        Shape ``(M, d)``: points in the open domain.
    boundary
        Shape ``(K, d)``: points on the boundary; ``K`` may be 0.
    target_boundary
        Shape ``(J, d)``: points on the boundary of the target set of a
        :class:`~maesolvers.core.boundary.Transport` problem, for a loss on
        the transport condition; empty by default. Any empty array is
        stored with shape ``(0, d)``.

    Raises
    ------
    ValueError
        If the arrays are not two-dimensional with the same number of columns,
        or ``interior`` is empty.
    """

    interior: np.ndarray
    boundary: np.ndarray
    target_boundary: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def __post_init__(self) -> None:
        interior = _read_only(self.interior)
        if interior.ndim != 2 or interior.shape[0] == 0:
            raise ValueError(
                f"Collocation.interior must have shape (M, d), M >= 1, "
                f"got {interior.shape}"
            )
        dim = interior.shape[1]
        target = _read_only(self.target_boundary)
        arrays = {
            "boundary": _read_only(self.boundary),
            "target_boundary": target.reshape(0, dim) if target.size == 0 else target,
        }
        for name, array in arrays.items():
            if array.ndim != 2 or array.shape[1] != dim:
                raise ValueError(
                    f"Collocation.{name} must have shape (K, {dim}), got {array.shape}"
                )
            object.__setattr__(self, name, array)
        object.__setattr__(self, "interior", interior)

    def fingerprint(self) -> str:
        """Return the SHA-256 of the points, to check that two solves share them.

        Returns
        -------
        str
            Hexadecimal digest of the interior points, the boundary points
            and the target boundary points, in this order, as little-endian
            float64 bytes; no target boundary points add no bytes, so a set
            without them has the digest it had before the field existed.
        """
        digest = hashlib.sha256()
        for points in (self.interior, self.boundary, self.target_boundary):
            digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
        return digest.hexdigest()


def _read_only(array: object) -> np.ndarray:
    """Return a read-only float copy of ``array``."""
    out = np.array(array, dtype=float)
    out.setflags(write=False)
    return out


def sample_interior(domain: Domain, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` uniform points in the open domain by rejection from its bounding box.

    Parameters
    ----------
    domain
        Any domain.
    n
        Number of points, at least 1.
    rng
        The random generator; the points depend on it alone.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, d)``, in the order they were accepted.

    Raises
    ------
    ValueError
        If fewer than ``n`` points are accepted after many rounds, as for a
        domain with no interior.
    """
    box = domain.bbox
    accepted: list[np.ndarray] = []
    count = 0
    for _ in range(_REJECTION_ROUNDS):
        draw = rng.uniform(box.lo, box.hi, size=(max(2 * (n - count), 64), domain.dim))
        inside = draw[domain.interior(draw)]
        accepted.append(inside)
        count += len(inside)
        if count >= n:
            return np.concatenate(accepted)[:n]
    raise ValueError(
        f"rejection sampling found {count} of {n} interior points of "
        f"{type(domain).__name__}"
    )


class Sampler(ABC):
    """Base class of every sampler (ADR 0004).

    A subclass is decorated with ``@dataclass(frozen=True, kw_only=True)``;
    its fields are its hyperparameters, so a solver's
    :attr:`~maesolvers.core.solver.Solver.hparams` record it and jsonargparse
    builds it from a ``class_path`` (ADR 0005). It implements
    :meth:`sample`, and overrides :meth:`supports` if it does not handle
    every domain.
    """

    def supports(self, problem: Problem) -> bool:
        """Whether the sampler can draw points for the problem's domain; True by default.

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
    def sample(self, problem: Problem, rng: np.random.Generator) -> Collocation:
        """Draw the collocation points of a problem.

        Parameters
        ----------
        problem
            A problem the sampler supports.
        rng
            The random generator, seeded by the solver; the points must
            depend on it and the sampler's fields alone.

        Returns
        -------
        Collocation
        """


@dataclass(frozen=True, kw_only=True)
class UniformSampler(Sampler):
    """Uniform random points inside the domain and on its boundary.

    The interior points come from rejection sampling in the bounding box
    (:func:`sample_interior`), the boundary points from the domain's
    :meth:`~maesolvers.core.domains.BoundarySampling.sample_boundary`, each
    piece of the boundary getting points in proportion to its length, as in
    Nystrom-Vestberg 2023, Section 4.

    Parameters
    ----------
    interior_points
        Number of points inside the domain, at least 1.
    boundary_points
        Number of points on the boundary, at least 0. With 0 the sampler
        supports any domain.

    Raises
    ------
    TypeError, ValueError
        On construction, for a count that is not an integer or is too small.
    """

    interior_points: int = 3000
    boundary_points: int = 400

    def __post_init__(self) -> None:
        for name, minimum in (("interior_points", 1), ("boundary_points", 0)):
            value = integer(name, getattr(self, name), minimum)
            object.__setattr__(self, name, value)

    def supports(self, problem: Problem) -> bool:
        """Whether the domain samples its boundary, or no boundary points are wanted.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        return self.boundary_points == 0 or isinstance(problem.domain, BoundarySampling)

    def sample(self, problem: Problem, rng: np.random.Generator) -> Collocation:
        """Draw the interior points, then the boundary points, from ``rng``.

        Parameters
        ----------
        problem
            A problem the sampler supports.
        rng
            The random generator.

        Returns
        -------
        Collocation
        """
        domain = problem.domain
        interior = sample_interior(domain, self.interior_points, rng)
        if self.boundary_points == 0:
            boundary = np.empty((0, domain.dim))
        else:
            assert isinstance(domain, BoundarySampling)  # supports checked it
            boundary = domain.sample_boundary(self.boundary_points, rng)
        return Collocation(interior=interior, boundary=boundary)


@dataclass(frozen=True, kw_only=True)
class TransportSampler(Sampler):
    """Uniform random points inside the domain, on its boundary and on the target's.

    For a :class:`~maesolvers.core.boundary.Transport` problem: the interior
    and boundary points as :class:`UniformSampler` draws them, in the same
    order from the same generator, then ``target_points`` points on the
    boundary of the target set, from its
    :meth:`~maesolvers.core.domains.BoundarySampling.sample_boundary`, for a
    loss that compares ``grad u`` on the boundary with the target's
    boundary (:class:`~maesolvers.neural.losses.HausdorffLoss`). The
    defaults are the 800 and 800 points of Caboussat-Peruso 2025 (Section
    4.1), and as many target points, which the paper does not give
    separately.

    Parameters
    ----------
    interior_points
        Number of points inside the domain, at least 1.
    boundary_points
        Number of points on the domain's boundary, at least 1.
    target_points
        Number of points on the target's boundary, at least 1.

    Raises
    ------
    TypeError, ValueError
        On construction, for a count that is not an integer or is less than 1.
    """

    interior_points: int = 800
    boundary_points: int = 800
    target_points: int = 800

    def __post_init__(self) -> None:
        for name in ("interior_points", "boundary_points", "target_points"):
            object.__setattr__(self, name, integer(name, getattr(self, name), 1))

    def supports(self, problem: Problem) -> bool:
        """Whether the problem is Transport and both its domain and target sample their boundary.

        Parameters
        ----------
        problem
            Any problem.

        Returns
        -------
        bool
        """
        boundary = problem.boundary
        return (
            isinstance(boundary, Transport)
            and isinstance(problem.domain, BoundarySampling)
            and isinstance(boundary.target, BoundarySampling)
        )

    def sample(self, problem: Problem, rng: np.random.Generator) -> Collocation:
        """Draw the interior points, the boundary points, then the target's, from ``rng``.

        Parameters
        ----------
        problem
            A problem the sampler supports.
        rng
            The random generator.

        Returns
        -------
        Collocation
            With ``target_boundary`` set.
        """
        domain, boundary = problem.domain, problem.boundary
        assert isinstance(domain, BoundarySampling)  # supports checked it
        assert isinstance(boundary, Transport)
        assert isinstance(boundary.target, BoundarySampling)
        interior = sample_interior(problem.domain, self.interior_points, rng)
        edge = domain.sample_boundary(self.boundary_points, rng)
        target = boundary.target.sample_boundary(self.target_points, rng)
        return Collocation(interior=interior, boundary=edge, target_boundary=target)
