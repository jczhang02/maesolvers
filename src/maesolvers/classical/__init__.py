"""Classical solvers: a finite-difference scheme composed with an iteration (ADR 0004).

``Newton(scheme=Standard9(n=64))`` and ``PoissonIteration(scheme=...)`` are
the parts; :class:`FDNewton` and :class:`FDPoisson` are the presets for the
standard 9-point scheme, :class:`FDMonotone` for the monotone wide-stencil
scheme :class:`Monotone17`, :class:`FDHybrid` for the hybrid scheme
:class:`Hybrid` that averages the two, :class:`FDFiltered` for the
filtered scheme :class:`Filtered`, which filters one with the other,
:class:`FDMultigrid` for the standard scheme on a 2D or 3D box
:class:`StandardBox` under the full multigrid iteration :class:`Multigrid`,
:class:`FDTransport` for :class:`SecondBoundary`, the transport problem with
its boundary condition on a square grid, :class:`FDMixed` for the monotone
mixed scheme :class:`MixedHJB` of Chen-Wan-Lin 2018, solved by
:class:`PolicyIteration`, and :class:`FDSemiLagrangian` for the
semi-Lagrangian scheme :class:`SemiLagrangian` under Howard's algorithm
:class:`Howard`. A new scheme subclasses :class:`Scheme`, a new iteration
:class:`SchemeIteration`; :class:`BoxGrid` is the grid of ``n`` cells per
side over a box in any dimension, with its operators in
:mod:`maesolvers.classical.grid`.
Everything here needs numpy and scipy only.
"""

from maesolvers.classical.filtered import Filtered
from maesolvers.classical.grid import BoxGrid
from maesolvers.classical.hybrid import Hybrid
from maesolvers.classical.iterations import (
    Howard,
    Newton,
    PoissonIteration,
    PolicyIteration,
    SchemeIteration,
)
from maesolvers.classical.mixed import MixedHJB
from maesolvers.classical.monotone import Monotone17
from maesolvers.classical.multigrid import Multigrid
from maesolvers.classical.presets import (
    FDFiltered,
    FDHybrid,
    FDMixed,
    FDMonotone,
    FDMultigrid,
    FDNewton,
    FDPoisson,
    FDSemiLagrangian,
    FDTransport,
)
from maesolvers.classical.schemes import Scheme, Standard9
from maesolvers.classical.second_boundary import SecondBoundary
from maesolvers.classical.semilagrangian import SemiLagrangian
from maesolvers.classical.standard_box import StandardBox

__all__ = [
    "BoxGrid",
    "FDFiltered",
    "FDHybrid",
    "FDMixed",
    "FDMonotone",
    "FDMultigrid",
    "FDNewton",
    "FDPoisson",
    "FDSemiLagrangian",
    "FDTransport",
    "Filtered",
    "Howard",
    "Hybrid",
    "MixedHJB",
    "Monotone17",
    "Multigrid",
    "Newton",
    "PoissonIteration",
    "PolicyIteration",
    "Scheme",
    "SchemeIteration",
    "SecondBoundary",
    "SemiLagrangian",
    "Standard9",
    "StandardBox",
]
