"""Core types: domains, boundary conditions, problems, results and the solver base.

Everything here needs numpy only. :mod:`maesolvers.metrics` evaluates a
:class:`Result`; the solvers live in :mod:`maesolvers.classical`.
:func:`integer`, :func:`positive`, :func:`nonnegative`, :func:`floats` and
:func:`boolean` check constructor parameters, and :func:`check_part` a part passed to a
solver or to another part, for solvers and parts of your own as for the
shipped ones. :func:`signed_distance`, :func:`support_function`,
:func:`closest_boundary_point`, :func:`volume`, :func:`integrate` and
:func:`check_mass_balance` serve the solvers of transport problems, with the
predicates :func:`knows_support_function` and :func:`can_check_mass_balance`;
:class:`SupportFunction` lets :func:`support_function` take a target of
your own.
"""

from maesolvers.core._params import boolean, floats, integer, nonnegative, positive
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
    SupportFunction,
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
from maesolvers.core.transport import (
    can_check_mass_balance,
    check_mass_balance,
    closest_boundary_point,
    integrate,
    knows_support_function,
    signed_distance,
    support_function,
    volume,
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
    "SupportFunction",
    "Trace",
    "Transport",
    "UnsupportedProblemError",
    "boolean",
    "can_check_mass_balance",
    "check_mass_balance",
    "check_part",
    "closest_boundary_point",
    "floats",
    "integer",
    "integrate",
    "knows_support_function",
    "nonnegative",
    "positive",
    "signed_distance",
    "support_function",
    "volume",
]
