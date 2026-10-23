"""The solvers maesolvers ships, by name.

:data:`SOLVERS` lists the presets (ADR 0004), the names the comparison tables
use. A solver of your own needs no registration: pass it to ``solve``
directly, or name it by its ``class_path`` in a config (ADR 0005).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from maesolvers.classical import (
    FDFiltered,
    FDHybrid,
    FDMonotone,
    FDNewton,
    FDPoisson,
    FDTransport,
)
from maesolvers.core.solver import Solver
from maesolvers.neural import (
    DirichletICNN,
    DirichletQSPINN,
    PenaltyMLP,
    TransportICNN,
    TransportQSPINN,
)

SOLVERS: Mapping[str, type[Solver]] = MappingProxyType(
    {
        cls.__name__: cls
        for cls in (
            FDNewton,
            FDPoisson,
            FDMonotone,
            FDHybrid,
            FDFiltered,
            FDTransport,
            PenaltyMLP,
            DirichletICNN,
            TransportICNN,
            DirichletQSPINN,
            TransportQSPINN,
        )
    }
)
"""Every preset by class name, read-only: the classical presets, then the
neural ones, each in the order they were added."""
