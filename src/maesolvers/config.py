"""Solvers from a configuration (ADR 0005).

A solver configuration names the solver by its ``class_path`` with its
``init_args``, nested parts in the same form:

.. code-block:: yaml

    class_path: Newton
    init_args:
      scheme:
        class_path: Standard9
        init_args: {n: 128}

A ``class_path`` is the class name of anything maesolvers ships, or the
dotted import path of any class, a solver of your own included.
``Result.hparams`` records a solver in this form, so
``from_config(result.hparams).solve(result.problem)`` reruns a result.
``maesolvers solve`` reads the same form under its ``solver`` key.

jsonargparse reads the configuration against the constructors' signatures,
so a misspelt key is an error and a value gets the type its parameter
declares. It is in the ``cli`` extra and imported only when a configuration
is read, never by ``import maesolvers``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from maesolvers import _jsonargparse
from maesolvers.core.solver import Solver


def from_config(config: Mapping[str, Any] | str | os.PathLike[str]) -> Solver:
    """Build a solver from its configuration.

    Parameters
    ----------
    config
        A mapping with the keys ``class_path`` and ``init_args``, as in the
        module docstring and in ``Result.hparams``, or the path of a YAML or
        JSON file that holds one. ``init_args`` may be absent or empty for a
        class whose parameters all have defaults.

    Returns
    -------
    Solver
        The solver, equal to the one its ``init_args`` build as keyword
        arguments.

    Raises
    ------
    ValueError
        For a missing or unknown key in any ``init_args``, a ``class_path``
        that does not import or is not a
        :class:`~maesolvers.core.solver.Solver`, or a value of the wrong
        type. The message names the key.
    TypeError, ValueError
        What a constructor raises for a value it rejects, as with keyword
        arguments.
    FileNotFoundError
        If ``config`` is a path and no file is there.
    ImportError
        If jsonargparse (the ``cli`` extra) is not installed.

    Examples
    --------
    >>> from maesolvers import T1
    >>> solver = from_config({"class_path": "FDNewton", "init_args": {"n": 32}})
    >>> result = solver.solve(T1)
    >>> from_config(result.hparams) == solver
    True
    """
    jsonargparse = _jsonargparse.module()
    if not isinstance(config, Mapping):
        config = _load(config)
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    _jsonargparse.add_solver_argument(parser)
    try:
        namespace = parser.parse_object({"solver": _empty_init_args(config)})
    except jsonargparse.ArgumentError as error:
        raise ValueError(f"invalid configuration: {error}") from None
    return parser.instantiate(namespace).solver


def _load(path: str | os.PathLike[str]) -> Any:
    """The content of a YAML or JSON file (JSON is YAML too)."""
    import yaml  # a dependency of jsonargparse

    if not os.path.isfile(path):
        raise FileNotFoundError(f"no configuration file {os.fspath(path)!r}")
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _empty_init_args(config: Any) -> Any:
    """``config`` with an ``init_args`` of None, at any depth, read as ``{}``.

    YAML reads ``init_args:`` with no value as None; jsonargparse would fail
    on it with a message that does not name the key.
    """
    if isinstance(config, Mapping):
        return {
            key: {}
            if key == "init_args" and value is None and "class_path" in config
            else _empty_init_args(value)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_empty_init_args(item) for item in config]
    return config
