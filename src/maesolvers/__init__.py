"""Solvers for the Monge-Ampere equation behind one interface.

Three concepts: a :class:`Problem` (domain, right-hand side, boundary
condition, exact solution when known), a :class:`Solver` whose
``solve(problem)`` returns a :class:`Result` (the solution as a callable plus,
for grid methods, the nodes, values and discrete Hessian), and
:func:`maesolvers.metrics.evaluate`, which measures any result the same way.
:func:`from_config` builds a solver from its configuration
(it needs the ``cli`` extra). The neural solvers of :mod:`maesolvers.neural`
need the ``neural`` extra to solve, not to be imported.
"""

from maesolvers.catalogue import (
    CATALOGUE,
    SHARED_TEST_SET,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T12,
    T16,
    T20,
    T22,
)
from maesolvers.classical import (
    FDNewton,
    FDPoisson,
    Newton,
    PoissonIteration,
    Scheme,
    SchemeIteration,
    Standard9,
)
from maesolvers.config import from_config
from maesolvers.core import (
    Box,
    Callback,
    CutQuarterDisk,
    Dirichlet,
    Disk,
    Ellipse,
    IterativeSolver,
    Problem,
    Result,
    Solution,
    Solver,
    Trace,
    Transport,
    UnsupportedProblemError,
)
from maesolvers.neural import (
    BFGS,
    ICNN,
    LBFGS,
    MLP,
    Adam,
    DirichletICNN,
    HardDirichlet,
    NeuralSolver,
    PenaltyLoss,
    PenaltyMLP,
    UniformSampler,
)
from maesolvers.registry import SOLVERS

__all__ = [
    "BFGS",
    "CATALOGUE",
    "ICNN",
    "LBFGS",
    "MLP",
    "SHARED_TEST_SET",
    "SOLVERS",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T12",
    "T16",
    "T20",
    "T22",
    "Adam",
    "Box",
    "Callback",
    "CutQuarterDisk",
    "Dirichlet",
    "DirichletICNN",
    "Disk",
    "Ellipse",
    "FDNewton",
    "FDPoisson",
    "HardDirichlet",
    "IterativeSolver",
    "NeuralSolver",
    "Newton",
    "PenaltyLoss",
    "PenaltyMLP",
    "PoissonIteration",
    "Problem",
    "Result",
    "Scheme",
    "SchemeIteration",
    "Solution",
    "Solver",
    "Standard9",
    "Trace",
    "Transport",
    "UniformSampler",
    "UnsupportedProblemError",
    "from_config",
]
