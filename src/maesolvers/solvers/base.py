"""The solver interface."""

from __future__ import annotations

from typing import Protocol

from maesolvers.problems import Problem
from maesolvers.result import Result


class Solver(Protocol):
    def solve(self, problem: Problem) -> Result: ...
