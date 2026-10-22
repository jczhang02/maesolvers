"""Solvers for the Monge-Ampere equation behind one interface.

Three concepts: a :class:`Problem` (domain, right-hand side, boundary data,
exact solution when known), a solver with a ``solve(problem) -> Result``
method, and a :class:`Result` (the solution as a callable plus the method's
own nodes, values and discrete Hessian). :mod:`maesolvers.metrics` evaluates
a result with the shared protocol.
"""

from maesolvers.domains import Box, Disk, Ellipse
from maesolvers.problems import CATALOGUE, P1, P2, P3, P4, P5, Problem
from maesolvers.result import Result
from maesolvers.solvers import FDNewton

__all__ = [
    "Box",
    "Disk",
    "Ellipse",
    "CATALOGUE",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "Problem",
    "Result",
    "FDNewton",
]
