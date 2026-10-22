"""What every solver returns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from maesolvers.problems import Problem


@dataclass
class Result:
    """Solution of a :class:`Problem` by one method.

    ``u`` evaluates the solution anywhere in the domain (an interpolant for
    grid methods, the network for neural methods). ``nodes`` are the points
    where the method itself defines the solution, ``values`` the solution
    there, and ``hessian`` the method's own discrete Hessian at those nodes,
    ``NaN`` where the method does not define one (grid boundary nodes).
    """

    problem: Problem
    u: Callable[[np.ndarray], np.ndarray]
    nodes: np.ndarray
    values: np.ndarray
    hessian: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
