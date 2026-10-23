"""Classical solvers: a finite-difference scheme composed with an iteration (ADR 0004).

``Newton(scheme=Standard9(n=64))`` and ``PoissonIteration(scheme=...)`` are
the parts; :class:`FDNewton` and :class:`FDPoisson` are the presets for the
standard 9-point scheme, :class:`FDMonotone` for the monotone wide-stencil
scheme :class:`Monotone17`, :class:`FDHybrid` for the hybrid scheme
:class:`Hybrid` that averages the two, and :class:`FDFiltered` for the
filtered scheme :class:`Filtered`, which filters one with the other, and
:class:`FDMultigrid` for the standard scheme on a 2D or 3D box
:class:`StandardBox` under the full multigrid iteration :class:`Multigrid`. A new
scheme subclasses :class:`Scheme`, a new iteration :class:`SchemeIteration`;
:class:`BoxGrid` is the grid of ``n`` cells per side over a box in any
dimension, with its operators in :mod:`maesolvers.classical.grid`.
Everything here needs numpy and scipy only.
"""

from maesolvers.classical.filtered import Filtered
from maesolvers.classical.grid import BoxGrid
from maesolvers.classical.hybrid import Hybrid
from maesolvers.classical.iterations import Newton, PoissonIteration, SchemeIteration
from maesolvers.classical.monotone import Monotone17
from maesolvers.classical.multigrid import Multigrid
from maesolvers.classical.presets import (
    FDFiltered,
    FDHybrid,
    FDMonotone,
    FDMultigrid,
    FDNewton,
    FDPoisson,
)
from maesolvers.classical.schemes import Scheme, Standard9
from maesolvers.classical.standard_box import StandardBox

__all__ = [
    "BoxGrid",
    "FDFiltered",
    "FDHybrid",
    "FDMonotone",
    "FDMultigrid",
    "FDNewton",
    "FDPoisson",
    "Filtered",
    "Hybrid",
    "Monotone17",
    "Multigrid",
    "Newton",
    "PoissonIteration",
    "Scheme",
    "SchemeIteration",
    "Standard9",
    "StandardBox",
]
