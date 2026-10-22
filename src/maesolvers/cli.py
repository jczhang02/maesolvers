"""The ``maesolvers`` command line (ADR 0005).

``maesolvers solve`` builds a solver from its configuration, as
:func:`maesolvers.from_config` does, picks a catalogue problem by its T id,
solves it and prints :func:`maesolvers.metrics.evaluate` as a table::

    maesolvers solve --config exp.yaml --solver.scheme.n 128 --problem T2
    maesolvers solve --solver FDNewton --solver.n 64 --problem T1 --output t1.json

A configuration file for ``--config`` has the keys ``solver``, a solver
configuration as :func:`maesolvers.from_config` reads it, ``problem``, a T id,
and optionally ``output``. Arguments apply in order, so an override after
``--config`` changes the file's value. ``--print_config`` prints the full
configuration without solving, as a file ``--config`` reads back.

``--output`` writes the run as strict JSON: ``run``, the configuration that
reruns it (``{"solver": Result.hparams, "problem": id}``, a ``--config``
file, and ``from_config(run["solver"])`` builds its solver), ``metrics``, the
fields of ``evaluate``, and ``info``, ``Result.info``. numpy scalars and
arrays are written as numbers and lists, and NaN and infinity as null.
``maesolvers solve --help`` lists the options and ``--solver.help FDNewton``
a solver's parameters.

Needs the ``cli`` extra (jsonargparse); ``python -m maesolvers`` runs the same
command.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from maesolvers import _jsonargparse
from maesolvers.catalogue import CATALOGUE
from maesolvers.core.result import Result
from maesolvers.core.solver import UnsupportedProblemError
from maesolvers.metrics import evaluate

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser


def _parser() -> ArgumentParser:
    """The ``maesolvers`` parser with its ``solve`` subcommand."""
    jsonargparse = _jsonargparse.module()
    solve = jsonargparse.ArgumentParser(
        description="Solve a catalogue problem and print evaluate as a table."
    )
    solve.add_argument(
        "--config",
        action=jsonargparse.ActionConfigFile,
        help="A YAML or JSON file with the keys solver, problem and output.",
    )
    _jsonargparse.add_solver_argument(solve)
    solve.add_argument(
        "--problem",
        choices=list(CATALOGUE),
        required=True,
        help="The T id of a catalogue problem.",
    )
    solve.add_argument(
        "--output",
        type=jsonargparse.typing.Path_fc | None,
        help="Write the run config, the metrics and Result.info as JSON to this path.",
    )
    parser = jsonargparse.ArgumentParser(
        prog="maesolvers",
        description="Solvers for the Monge-Ampere equation det D^2 u = f.",
    )
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("solve", solve, help=solve.description)
    return parser


def _table(result: Result, metrics: Mapping[str, float]) -> str:
    """A status line, then one ``name value`` row per metric."""
    info = result.info
    status = "converged" if info["converged"] else "did not converge"
    lines = [
        f"{_solver_name(result)} on {result.problem.name}: "
        f"{status} after {info['iterations']} iterations"
    ]
    width = max(map(len, metrics))
    for name, value in metrics.items():
        text = str(value) if isinstance(value, int) else f"{value:.6g}"
        lines.append(f"{name:<{width}}  {text}")
    return "\n".join(lines)


def _solver_name(result: Result) -> str:
    """The solver's class name, from ``result.hparams``."""
    return str(result.hparams["class_path"]).rpartition(".")[2]


def _jsonable(value: Any) -> Any:
    """``value`` in the types of strict JSON.

    numpy scalars become Python scalars, arrays and tuples lists, and a NaN
    or infinite float None, at any depth; anything else that JSON cannot
    hold is left for :func:`json.dumps` to reject.
    """
    if isinstance(value, np.ndarray | np.generic):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``maesolvers`` command.

    Parameters
    ----------
    argv
        The arguments after the program name; ``sys.argv[1:]`` if None.

    Raises
    ------
    SystemExit
        With status 2 and a message on standard error for a wrong argument,
        an ``--output`` path that cannot be created, a value a constructor
        rejects with ValueError or a problem the solver does not support;
        with status 0 after ``--help``.
    """
    parser = _parser()
    namespace = parser.parse_args(argv)
    try:
        run = parser.instantiate(namespace).solve
    except ValueError as error:
        parser.error(f"solve: {error}")
    try:
        result = run.solver.solve(CATALOGUE[run.problem])
    except UnsupportedProblemError as error:
        parser.error(f"solve: {error}")
    metrics = evaluate(result)
    print(_table(result, metrics))
    if run.output is not None:
        record = {
            "run": {"solver": result.hparams, "problem": result.problem.name},
            "metrics": metrics,
            "info": result.info,
        }
        text = json.dumps(_jsonable(record), indent=2, allow_nan=False)
        Path(run.output).write_text(text + "\n")
