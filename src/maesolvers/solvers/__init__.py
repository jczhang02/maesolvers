"""Solvers. Each subclasses :class:`~maesolvers.core.solver.Solver`."""

from maesolvers.solvers.fd_newton import FDNewton
from maesolvers.solvers.fd_poisson import FDPoisson

__all__ = ["FDNewton", "FDPoisson"]
