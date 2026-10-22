"""The solvers maesolvers ships, by name.

:data:`SOLVERS` lists the presets (ADR 0004), the names the comparison tables
use. A solver of your own needs no registration: pass it to ``solve``
directly, or name it by its ``class_path`` in a config (ADR 0005).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from maesolvers.classical import FDNewton, FDPoisson
from maesolvers.core.solver import Solver
from maesolvers.neural import DirichletICNN, PenaltyMLP

SOLVERS: Mapping[str, type[Solver]] = MappingProxyType(
    {cls.__name__: cls for cls in (FDNewton, FDPoisson, PenaltyMLP, DirichletICNN)}
)
"""Every preset by class name, read-only."""
