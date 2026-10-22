"""The jsonargparse pieces that :mod:`maesolvers.config` and :mod:`maesolvers.cli` share.

jsonargparse is in the ``cli`` extra (ADR 0006), so it is imported here only
when a configuration is read, never by ``import maesolvers``.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

from maesolvers.core.solver import Solver

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser


def module() -> types.ModuleType:
    """Import jsonargparse, or raise an ImportError that names the extra to install.

    Returns
    -------
    types.ModuleType
        :mod:`jsonargparse`.

    Raises
    ------
    ImportError
        If jsonargparse is not installed.
    """
    try:
        import jsonargparse
    except ImportError as error:
        raise ImportError(
            "configurations need jsonargparse, in the cli extra: "
            "pip install 'maesolvers[cli]'"
        ) from error
    return jsonargparse


def add_solver_argument(parser: ArgumentParser) -> None:
    """Add ``--solver``, a solver as a ``class_path`` with its ``init_args``.

    The one definition of a solver configuration, so :func:`maesolvers.from_config`
    and ``maesolvers solve`` read it the same way.

    Parameters
    ----------
    parser
        A jsonargparse parser.
    """
    parser.add_argument(
        "--solver",
        type=Solver,
        required=True,
        help="The solver: a class_path with its init_args.",
    )
