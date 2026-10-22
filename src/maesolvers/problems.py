"""The catalogue P1 to P5, provisional.

It moves to the T ids of the shared test-set specification in the next step
of 0.0.2 (ADR 0007); numbers are never reused.
"""

from __future__ import annotations

import numpy as np

from maesolvers.core.boundary import Dirichlet, Transport
from maesolvers.core.domains import Box, Disk, Ellipse, Field
from maesolvers.core.problem import Problem


def _r2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[..., 0] ** 2 + x[..., 1] ** 2


def _const(value: float) -> Field:
    return lambda x: np.full(np.shape(x)[:-1], float(value))


P1 = Problem(
    "P1",
    Disk((0.0, 0.0), 1.0),
    f=_const(1.0),
    boundary=Dirichlet(_const(0.5)),
    exact=lambda x: 0.5 * _r2(x),
    note="Branch selection: the concave 1 - |x|^2/2 also satisfies the equation.",
)

P2 = Problem(
    "P2",
    Box((-1.0, -1.0), (1.0, 1.0)),
    f=lambda x: (1.0 + _r2(x)) ** -2,
    boundary=Dirichlet(lambda x: np.sqrt(1.0 + _r2(x))),
    exact=lambda x: np.sqrt(1.0 + _r2(x)),
    note="Smooth non-quadratic solution; the Hessian varies with x.",
)

P3 = Problem(
    "P3",
    Disk((0.0, 0.0), 1.0),
    f=_const(1.0),
    boundary=Transport(Ellipse((3.5, 0.0), (2.0, 0.5))),
    exact=lambda x: x[..., 0] ** 2 + 3.5 * x[..., 0] + 0.25 * x[..., 1] ** 2,
    note=(
        "Uniform source density f = 1 on the unit disk to the uniform target "
        "density on the ellipse, both of area pi (Caboussat-Peruso 2025)."
    ),
)

P4 = Problem(
    "P4",
    Box((0.0, 0.0), (1.0, 1.0)),
    f=lambda x: (1.0 + _r2(x)) * np.exp(_r2(x)),
    boundary=Dirichlet(lambda x: np.exp(0.5 * _r2(x))),
    exact=lambda x: np.exp(0.5 * _r2(x)),
    note="Smooth solution, standard example of the finite-difference literature.",
)

P5 = Problem(
    "P5",
    Box((0.0, 0.0), (1.0, 1.0)),
    f=lambda x: 2.0 * (2.0 - _r2(x)) ** -2,
    boundary=Dirichlet(lambda x: -np.sqrt(2.0 - _r2(x))),
    exact=lambda x: -np.sqrt(2.0 - _r2(x)),
    note="Gradient blows up at the corner (1, 1); u is not C^1 on the closed square.",
)

CATALOGUE: dict[str, Problem] = {p.name: p for p in (P1, P2, P3, P4, P5)}
