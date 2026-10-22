"""Classical solvers: a finite-difference scheme composed with an iteration (ADR 0004).

``Newton(scheme=Standard9(n=64))`` and ``PoissonIteration(scheme=...)`` are
the parts; :class:`FDNewton` and :class:`FDPoisson` are the presets for the
standard 9-point scheme, :class:`FDMonotone` for the monotone wide-stencil
scheme :class:`Monotone17`. A new scheme subclasses :class:`Scheme`, a new
iteration :class:`SchemeIteration`. Everything here needs numpy and scipy
only.
"""

from maesolvers.classical.iterations import Newton, PoissonIteration, SchemeIteration
from maesolvers.classical.monotone import Monotone17
from maesolvers.classical.presets import FDMonotone, FDNewton, FDPoisson
from maesolvers.classical.schemes import Scheme, Standard9

__all__ = [
    "FDMonotone",
    "FDNewton",
    "FDPoisson",
    "Monotone17",
    "Newton",
    "PoissonIteration",
    "Scheme",
    "SchemeIteration",
    "Standard9",
]
