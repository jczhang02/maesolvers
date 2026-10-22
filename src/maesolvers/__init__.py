"""Solvers for the Monge-Ampere equation behind one interface.

Three concepts: a :class:`Problem` (domain, right-hand side, boundary
condition, exact solution when known), a :class:`Solver` whose
``solve(problem)`` returns a :class:`Result` (the solution as a callable plus,
for grid methods, the nodes, values and discrete Hessian), and
:func:`maesolvers.metrics.evaluate`, which measures any result the same way.
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
from maesolvers.solvers import FDNewton, FDPoisson

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
    "Box",
    "Callback",
    "CutQuarterDisk",
    "Dirichlet",
    "Disk",
    "Ellipse",
    "FDNewton",
    "FDPoisson",
    "IterativeSolver",
    "Problem",
    "Result",
    "Solution",
    "Solver",
    "Trace",
    "Transport",
    "UnsupportedProblemError",
]
