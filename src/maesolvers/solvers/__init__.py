"""Solvers. Each exposes ``solve(problem) -> Result``."""

from maesolvers.solvers.fd_newton import FDNewton
from maesolvers.solvers.fd_poisson import FDPoisson

__all__ = ["FDNewton", "FDPoisson"]
