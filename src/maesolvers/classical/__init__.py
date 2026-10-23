"""Classical solvers: a finite-difference scheme composed with an iteration (ADR 0004).

``Newton(scheme=Standard9(n=64))`` and ``PoissonIteration(scheme=...)`` are
the parts; :class:`FDNewton` and :class:`FDPoisson` are the presets for the
standard 9-point scheme, :class:`FDMonotone` for the monotone wide-stencil
scheme :class:`Monotone17`, :class:`FDHybrid` for the hybrid scheme
:class:`Hybrid` that averages the two, and :class:`FDFiltered` for the
filtered scheme :class:`Filtered`, which filters one with the other. A new
scheme subclasses :class:`Scheme`, a new iteration :class:`SchemeIteration`.
Everything here needs numpy and scipy only.
"""

from maesolvers.classical.filtered import Filtered
from maesolvers.classical.hybrid import Hybrid
from maesolvers.classical.iterations import Newton, PoissonIteration, SchemeIteration
from maesolvers.classical.monotone import Monotone17
from maesolvers.classical.presets import (
    FDFiltered,
    FDHybrid,
    FDMonotone,
    FDNewton,
    FDPoisson,
)
from maesolvers.classical.schemes import Scheme, Standard9

__all__ = [
    "FDFiltered",
    "FDHybrid",
    "FDMonotone",
    "FDNewton",
    "FDPoisson",
    "Filtered",
    "Hybrid",
    "Monotone17",
    "Newton",
    "PoissonIteration",
    "Scheme",
    "SchemeIteration",
    "Standard9",
]
