"""Core types: domains, boundary conditions, problems, results and the solver base.

Everything here needs numpy only. :mod:`maesolvers.metrics` evaluates a
:class:`Result`; the solvers live in :mod:`maesolvers.classical`.
:func:`integer`, :func:`positive`, :func:`nonnegative` and :func:`boolean`
check constructor parameters, and :func:`check_part` a part passed to a
solver or to another part, for solvers and parts of your own as for the
shipped ones.
"""

from maesolvers.core._params import boolean, integer, nonnegative, positive
from maesolvers.core.boundary import BoundaryCondition, Dirichlet, Transport
from maesolvers.core.callbacks import Callback
from maesolvers.core.domains import (
    BoundaryDistance,
    BoundarySampling,
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
    check_part,
)

__all__ = [
    "BoundaryCondition",
    "BoundaryDistance",
    "BoundarySampling",
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
    "boolean",
    "check_part",
    "integer",
    "nonnegative",
    "positive",
]
