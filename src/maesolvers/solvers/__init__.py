"""Solvers. Each exposes ``solve(problem) -> Result``."""

from maesolvers.solvers.fd_newton import FDNewton

__all__ = ["FDNewton"]
