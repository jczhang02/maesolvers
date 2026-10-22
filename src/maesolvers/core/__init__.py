"""Core types: domains, boundary conditions, problems, results and the solver base.

Everything here needs numpy only. :mod:`maesolvers.metrics` evaluates a
:class:`Result`; the solvers live in :mod:`maesolvers.solvers`.
"""

from maesolvers.core.boundary import BoundaryCondition, Dirichlet, Transport
from maesolvers.core.callbacks import Callback
from maesolvers.core.domains import (
    Box,
    CutQuarterDisk,
    Disk,
    Domain,
    Ellipse,
    Field,
    HessianField,
)
from maesolvers.core.problem import Observable, Problem
from maesolvers.core.result import Result
from maesolvers.core.solver import (
    IterativeSolver,
    Solution,
    Solver,
    Trace,
    UnsupportedProblemError,
)

__all__ = [
    "BoundaryCondition",
    "Box",
    "Callback",
    "CutQuarterDisk",
    "Dirichlet",
    "Disk",
    "Domain",
    "Ellipse",
    "Field",
    "HessianField",
    "IterativeSolver",
    "Observable",
    "Problem",
    "Result",
    "Solution",
    "Solver",
    "Trace",
    "Transport",
    "UnsupportedProblemError",
]
