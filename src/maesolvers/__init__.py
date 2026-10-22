"""Solvers for the Monge-Ampere equation behind one interface.

Three concepts: a :class:`Problem` (domain, right-hand side, boundary
condition, exact solution when known), a :class:`Solver` whose
``solve(problem)`` returns a :class:`Result` (the solution as a callable plus,
for grid methods, the nodes, values and discrete Hessian), and
:func:`maesolvers.metrics.evaluate`, which measures any result the same way.
:func:`from_config` builds a solver and picks a problem from a configuration
(it needs the ``cli`` extra).
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
from maesolvers.registry import SOLVERS

__all__ = [
    "CATALOGUE",
    "SHARED_TEST_SET",
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
    "SOLVERS",
    "Box",
    "Callback",
    "CutQuarterDisk",
    "Dirichlet",
    "Disk",
    "Ellipse",
    "FDNewton",
    "FDPoisson",
    "IterativeSolver",
    "Newton",
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
    "UnsupportedProblemError",
    "from_config",
]
