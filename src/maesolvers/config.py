"""Solvers and problems from a configuration (ADR 0005).

A configuration names a solver by its ``class_path`` with its ``init_args``,
nested parts in the same form, and a catalogue problem by its T id:

.. code-block:: yaml

    solver:
      class_path: Newton
      init_args:
        scheme:
          class_path: Standard9
          init_args: {n: 128}
    problem: T2

A ``class_path`` is the class name of anything maesolvers ships, or the
dotted import path of any class, a solver of your own included.
``Result.hparams`` records a solver in this form, so
``{"solver": result.hparams, "problem": result.problem.name}`` reruns a result.

jsonargparse reads the configuration against the constructors' signatures,
so a misspelt key is an error and a value gets the type its parameter
declares. It is in the ``cli`` extra and imported only when a configuration
is read, never by ``import maesolvers``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from maesolvers.catalogue import CATALOGUE
from maesolvers.core.problem import Problem
from maesolvers.core.solver import Solver

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser


def _add_run_arguments(parser: ArgumentParser) -> None:
    """Add the arguments of one run, ``--solver`` and ``--problem``, to a parser.

    The one definition of a configuration that :func:`from_config` and the
    ``maesolvers solve`` command share, so the two read it the same way.

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
    parser.add_argument(
        "--problem",
        choices=list(CATALOGUE),
        required=True,
        help="The T id of a catalogue problem.",
    )


def _jsonargparse() -> Any:
    """The jsonargparse module, or an ImportError that names the extra to install.

    Returns
    -------
    module
        :mod:`jsonargparse`.

    Raises
    ------
    FileNotFoundError
        If ``config`` is a path and no file is there.
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


def from_config(
    config: Mapping[str, Any] | str | os.PathLike[str],
) -> tuple[Solver, Problem]:
    """Build a solver and pick a catalogue problem from a configuration.

    Parameters
    ----------
    config
        A mapping with the keys ``solver`` and ``problem``, as in the module
        docstring, or the path of a YAML or JSON file that holds one.

    Returns
    -------
    solver : Solver
        The solver, equal to the one its ``init_args`` build as keyword
        arguments.
    problem : Problem
        ``CATALOGUE[config["problem"]]``.

    Raises
    ------
    ValueError
        For a missing or unknown key, at the top level or in any
        ``init_args``, a ``class_path`` that does not import or is not a
        :class:`~maesolvers.core.solver.Solver`, a value of the wrong type
        or an unknown problem id. The message names the key.
    TypeError, ValueError
        What a constructor raises for a value it rejects, as with keyword
        arguments.
    FileNotFoundError
        If ``config`` is a path and no file is there.
    ImportError
        If jsonargparse (the ``cli`` extra) is not installed.

    Examples
    --------
    >>> solver, problem = from_config(
    ...     {"solver": {"class_path": "FDNewton", "init_args": {"n": 32}},
    ...      "problem": "T1"}
    ... )
    >>> result = solver.solve(problem)
    """
    jsonargparse = _jsonargparse()
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    _add_run_arguments(parser)
    try:
        if isinstance(config, Mapping):
            namespace = parser.parse_object(dict(config))
        elif os.path.isfile(config):
            namespace = parser.parse_path(os.fspath(config))
        else:
            raise FileNotFoundError(f"no configuration file {os.fspath(config)!r}")
    except jsonargparse.ArgumentError as error:
        raise ValueError(f"invalid configuration: {error}") from None
    built = parser.instantiate(namespace)
    return built.solver, CATALOGUE[built.problem]
