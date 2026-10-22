"""Solvers for the Monge-Ampere equation behind one interface.

Three concepts: a :class:`Problem` (domain, right-hand side, boundary
condition, exact solution when known), a :class:`Solver` whose
``solve(problem)`` returns a :class:`Result` (the solution as a callable plus,
for grid methods, the nodes, values and discrete Hessian), and
:func:`maesolvers.metrics.evaluate`, which measures any result the same way.
"""

from maesolvers.classical import (
    FDNewton,
    FDPoisson,
    Newton,
    PoissonIteration,
    Scheme,
    SchemeIteration,
    Standard9,
)
from maesolvers.core import (
    Box,
    Callback,
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
from maesolvers.problems import CATALOGUE, P1, P2, P3, P4, P5
from maesolvers.registry import SOLVERS

__all__ = [
    "CATALOGUE",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "SOLVERS",
    "Box",
    "Callback",
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
]
