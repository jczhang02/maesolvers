"""Conversion of constructor parameters, with errors that name the parameter.

The shipped solvers and parts call these in ``__post_init__``; they are
exported from :mod:`maesolvers.core` so that a solver of your own checks its
parameters in the same way, with the same messages.
"""

from __future__ import annotations

import numbers
import operator


def integer(name: str, value: object, minimum: int) -> int:
    """Check an integer parameter and return it as a plain ``int``.

    Parameters
    ----------
    name : str
        The parameter's name, used in the error messages.
    value : object
        The value given; a Python or numpy integer.
    minimum : int
        The smallest value allowed.

    Returns
    -------
    int
        ``value`` as a plain ``int``, so that ``hparams`` and configurations
        hold no numpy scalars.

    Raises
    ------
    TypeError
        If ``value`` is a bool or is not an integer (a float or a str, say):
        ``"<name> must be an integer, got <value>"``.
    ValueError
        If ``value`` is less than ``minimum``:
        ``"<name> must be at least <minimum>, got <value>"``.

    Examples
    --------
    In the ``__post_init__`` of a frozen dataclass:

    >>> object.__setattr__(self, "n", integer("n", self.n, 2))  # doctest: +SKIP
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        out = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if out < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {out}")
    return out


def positive(name: str, value: object) -> float:
    """Check a positive, finite real parameter and return it as a plain ``float``.

    Parameters
    ----------
    name : str
        The parameter's name, used in the error messages.
    value : object
        The value given; a Python or numpy real number.

    Returns
    -------
    float
        ``value`` as a plain ``float``, so that ``hparams`` and configurations
        hold no numpy scalars.

    Raises
    ------
    TypeError
        If ``value`` is a bool or is not a real number (a str, say):
        ``"<name> must be a real number, got <value>"``.
    ValueError
        If ``value`` is zero, negative, infinite or NaN:
        ``"<name> must be positive and finite, got <value>"``.

    Examples
    --------
    In the ``__post_init__`` of a frozen dataclass:

    >>> object.__setattr__(self, "tol", positive("tol", self.tol))  # doctest: +SKIP
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not 0.0 < out < float("inf"):
        raise ValueError(f"{name} must be positive and finite, got {out}")
    return out
