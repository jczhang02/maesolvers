"""The problem definition: find convex ``u`` with ``det D^2 u = f``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType

from maesolvers.core.boundary import BoundaryCondition, Dirichlet, Transport
from maesolvers.core.domains import Domain, Field

Observable = Callable[[Field], float]
"""Maps a solution ``u`` (a :data:`Field`) to one number, such as ``u(0, 0)``."""


@dataclass(frozen=True, eq=False)
class Problem:
    """Find convex ``u`` with ``det D^2 u = f`` in ``domain``.

    Problems compare by identity: two problems with the same name are still
    different objects, because their callables cannot be compared. The fields
    after ``boundary`` are keyword-only.

    Parameters
    ----------
    name
        Catalogue id, unique within the catalogue.
    domain
        The domain, a :class:`~maesolvers.core.domains.Domain`.
    f
        Right-hand side, positive or zero; a :data:`Field`. For a
        :class:`~maesolvers.core.boundary.Transport` problem it is the source
        density; the target density is uniform 1 on the target.
    boundary
        :class:`~maesolvers.core.boundary.Dirichlet` or
        :class:`~maesolvers.core.boundary.Transport`.
    exact
        The convex exact solution when it is known; for a transport problem,
        any one of the solutions (they differ by a constant).
    observables
        Named numbers computed from a solution, reported by
        :func:`maesolvers.metrics.evaluate`; mainly for problems without an
        exact solution. Each name is an ASCII identifier, such as
        ``u_origin``, so that it is also a valid metric name for MLflow.
    note
        What the problem tests, in one or two sentences.
    """

    name: str
    domain: Domain
    f: Field
    boundary: BoundaryCondition
    _: KW_ONLY
    exact: Field | None = None
    observables: Mapping[str, Observable] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Problem.name must not be empty")
        if not isinstance(self.domain, Domain):
            raise TypeError(
                f"Problem.domain must be a Domain, got {type(self.domain).__name__}"
            )
        if not callable(self.f):
            raise TypeError("Problem.f must be callable")
        if not isinstance(self.boundary, (Dirichlet, Transport)):
            raise TypeError(
                "Problem.boundary must be Dirichlet or Transport, "
                f"got {type(self.boundary).__name__}"
            )
        if (
            isinstance(self.boundary, Transport)
            and self.boundary.target.dim != self.domain.dim
        ):
            raise ValueError(
                f"the transport target is {self.boundary.target.dim}D "
                f"but the domain is {self.domain.dim}D"
            )
        if self.exact is not None and not callable(self.exact):
            raise TypeError("Problem.exact must be callable or None")
        for key, observable in self.observables.items():
            if not (isinstance(key, str) and key.isascii() and key.isidentifier()):
                raise ValueError(
                    f"observable name {key!r} must be an ASCII identifier, "
                    "such as 'u_origin'"
                )
            if not callable(observable):
                raise TypeError(f"observable {key!r} must be callable")
        object.__setattr__(
            self, "observables", MappingProxyType(dict(self.observables))
        )

    @property
    def dim(self) -> int:
        """Space dimension of the domain."""
        return self.domain.dim

    def __repr__(self) -> str:
        return (
            f"Problem({self.name!r}, {self.domain!r}, {type(self.boundary).__name__})"
        )
