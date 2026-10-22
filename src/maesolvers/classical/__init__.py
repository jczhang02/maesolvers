"""Classical solvers: a finite-difference scheme composed with an iteration (ADR 0004).

``Newton(scheme=Standard9(n=64))`` and ``PoissonIteration(scheme=...)`` are
the parts; :class:`FDNewton` and :class:`FDPoisson` are the presets for the
standard 9-point scheme. A new scheme subclasses :class:`Scheme`, a new
iteration :class:`SchemeIteration`. Everything here needs numpy and scipy
only.
"""

from maesolvers.classical.iterations import Newton, PoissonIteration, SchemeIteration
from maesolvers.classical.presets import FDNewton, FDPoisson
from maesolvers.classical.schemes import Scheme, Standard9

__all__ = [
    "FDNewton",
    "FDPoisson",
    "Newton",
    "PoissonIteration",
    "Scheme",
    "SchemeIteration",
    "Standard9",
]
