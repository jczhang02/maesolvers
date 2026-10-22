"""Problem definition and the shared catalogue P1 to P5.

The catalogue follows the project's test-problem page: numbers are never
reused, and every solver reports on these problems.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from maesolvers.domains import Box, Disk, Ellipse

Field = Callable[[np.ndarray], np.ndarray]
"""Maps points of shape ``(..., 2)`` to values of shape ``(...)``."""


@dataclass(frozen=True)
class Problem:
    """Find convex ``u`` with ``det D^2 u = f`` in ``domain``.

    ``boundary`` is ``"dirichlet"`` (``u = g`` on the boundary) or
    ``"transport"`` (``grad u`` maps ``domain`` onto ``target``).
    """

    name: str
    domain: Box | Disk
    f: Field
    g: Field | None = None
    exact: Field | None = None
    boundary: str = "dirichlet"
    target: Ellipse | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.boundary not in ("dirichlet", "transport"):
            raise ValueError(f"unknown boundary type {self.boundary!r}")
        if self.boundary == "dirichlet" and self.g is None:
            raise ValueError("a Dirichlet problem needs boundary data g")
        if self.boundary == "transport" and self.target is None:
            raise ValueError("a transport problem needs a target set")


def _r2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[..., 0] ** 2 + x[..., 1] ** 2


def _const(value: float) -> Field:
    return lambda x: np.full(np.asarray(x).shape[:-1], float(value))


P1 = Problem(
    "P1",
    Disk((0.0, 0.0), 1.0),
    f=_const(1.0),
    g=_const(0.5),
    exact=lambda x: 0.5 * _r2(x),
    note="Branch selection: the concave 1 - |x|^2/2 also satisfies the equation.",
)

P2 = Problem(
    "P2",
    Box((-1.0, -1.0), (1.0, 1.0)),
    f=lambda x: (1.0 + _r2(x)) ** -2,
    g=lambda x: np.sqrt(1.0 + _r2(x)),
    exact=lambda x: np.sqrt(1.0 + _r2(x)),
    note="Smooth non-quadratic solution; the Hessian varies with x.",
)

P3 = Problem(
    "P3",
    Disk((0.0, 0.0), 1.0),
    f=_const(1.0),
    exact=lambda x: x[..., 0] ** 2 + 3.5 * x[..., 0] + 0.25 * x[..., 1] ** 2,
    boundary="transport",
    target=Ellipse((3.5, 0.0), (2.0, 0.5)),
    note=(
        "Uniform density on the unit disk to uniform density on the ellipse; "
        "f/g = 1 so the equation is det D^2 u = 1 (Caboussat-Peruso 2025)."
    ),
)

P4 = Problem(
    "P4",
    Box((0.0, 0.0), (1.0, 1.0)),
    f=lambda x: (1.0 + _r2(x)) * np.exp(_r2(x)),
    g=lambda x: np.exp(0.5 * _r2(x)),
    exact=lambda x: np.exp(0.5 * _r2(x)),
    note="Smooth solution, standard example of the finite-difference literature.",
)

P5 = Problem(
    "P5",
    Box((0.0, 0.0), (1.0, 1.0)),
    f=lambda x: 2.0 * (2.0 - _r2(x)) ** -2,
    g=lambda x: -np.sqrt(2.0 - _r2(x)),
    exact=lambda x: -np.sqrt(2.0 - _r2(x)),
    note="Gradient blows up at the corner (1, 1); u is not C^1 on the closed square.",
)

CATALOGUE: dict[str, Problem] = {p.name: p for p in (P1, P2, P3, P4, P5)}
