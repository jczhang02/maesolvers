"""Classical solvers: a finite-difference scheme composed with an iteration (ADR 0004).

``Newton(scheme=Standard9(n=64))`` and ``PoissonIteration(scheme=...)`` are
the parts; :class:`FDNewton` and :class:`FDPoisson` are the presets for the
standard 9-point scheme. Everything here needs numpy and scipy only.
"""

from maesolvers.classical.iterations import (
    Newton,
    NewtonScheme,
    PoissonIteration,
    PoissonScheme,
)
from maesolvers.classical.presets import FDNewton, FDPoisson
from maesolvers.classical.schemes import Grid, Scheme, Standard9

__all__ = [
    "FDNewton",
    "FDPoisson",
    "Grid",
    "Newton",
    "NewtonScheme",
    "PoissonIteration",
    "PoissonScheme",
    "Scheme",
    "Standard9",
]
