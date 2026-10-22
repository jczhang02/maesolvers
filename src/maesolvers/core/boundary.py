"""Boundary conditions.

A problem carries exactly one boundary condition, an instance of one of the
types below; solvers declare which types they support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from maesolvers.core.domains import Domain, Field


@dataclass(frozen=True)
class Dirichlet:
    """``u = g`` on the boundary of the domain.

    Parameters
    ----------
    g
        Boundary values, a :data:`Field`.
    """

    g: Field

    def __post_init__(self) -> None:
        if not callable(self.g):
            raise TypeError(
                f"Dirichlet.g must be callable, got {type(self.g).__name__}"
            )


@dataclass(frozen=True)
class Transport:
    """Second boundary value problem: ``grad u`` maps the domain onto ``target``.

    The solution is unique up to an additive constant.

    Parameters
    ----------
    target
        The target set, a :class:`~maesolvers.core.domains.Domain` of the same
        dimension as the problem's domain.
    """

    target: Domain

    def __post_init__(self) -> None:
        if not isinstance(self.target, Domain):
            raise TypeError(
                f"Transport.target must be a Domain, got {type(self.target).__name__}"
            )


BoundaryCondition = Union[Dirichlet, Transport]
"""Any boundary condition a :class:`~maesolvers.core.problem.Problem` accepts."""
