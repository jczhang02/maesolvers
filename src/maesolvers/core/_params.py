"""Conversion of constructor parameters, with errors that name the parameter.

The shipped solvers and parts call these in ``__post_init__``; they are
exported from :mod:`maesolvers.core` so that a solver of your own checks its
parameters in the same way, with the same messages. :func:`points`, the
shape check of an array of points, is shared by the domains and the grids
inside the package and is not exported.
"""

from __future__ import annotations

import numbers
import operator

import numpy as np


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


def nonnegative(name: str, value: object) -> float:
    """Check a nonnegative, finite real parameter and return it as a plain ``float``.

    Parameters
    ----------
    name : str
        The parameter's name, used in the error messages.
    value : object
        The value given; a Python or numpy real number.

    Returns
    -------
    float
        ``value`` as a plain ``float``.

    Raises
    ------
    TypeError
        If ``value`` is a bool or is not a real number:
        ``"<name> must be a real number, got <value>"``.
    ValueError
        If ``value`` is negative, infinite or NaN:
        ``"<name> must be nonnegative and finite, got <value>"``.

    Examples
    --------
    In the ``__post_init__`` of a frozen dataclass, for a weight that 0 turns off:

    >>> object.__setattr__(self, "weight", nonnegative("weight", self.weight))  # doctest: +SKIP
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not 0.0 <= out < float("inf"):
        raise ValueError(f"{name} must be nonnegative and finite, got {out}")
    return out


def floats(name: str, value: object) -> tuple[float, ...]:
    """Check a sequence of finite real numbers and return it as a tuple of floats.

    Parameters
    ----------
    name : str
        The parameter's name, used in the error messages.
    value : object
        The value given; a sequence (a tuple, a list, a numpy array) of
        Python or numpy real numbers.

    Returns
    -------
    tuple of float
        ``value`` as a tuple of plain ``float``, so that ``hparams`` and
        configurations hold no numpy scalars.

    Raises
    ------
    TypeError
        If ``value`` is a str or bytes, is not iterable, or holds a bool or
        an element that is not a real number:
        ``"<name> must be a sequence of numbers, got <value>"``.
    ValueError
        If ``value`` is empty: ``"<name> must hold one or more numbers"``; or
        if an element is infinite or NaN: ``"<name> must be finite, got
        <value>"``.

    Examples
    --------
    In the ``__post_init__`` of a frozen dataclass, for a point:

    >>> object.__setattr__(self, "x0", floats("x0", self.x0))  # doctest: +SKIP
    """
    error = TypeError(f"{name} must be a sequence of numbers, got {value!r}")
    if isinstance(value, (str, bytes)):
        raise error
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise error from None
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)
        for v in items
    ):
        raise error
    out = tuple(float(v) for v in items)
    if not out:
        raise ValueError(f"{name} must hold one or more numbers")
    if not all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def boolean(name: str, value: object) -> bool:
    """Check a bool parameter and return it as a plain ``bool``.

    Parameters
    ----------
    name : str
        The parameter's name, used in the error message.
    value : object
        The value given; a Python or numpy bool.

    Returns
    -------
    bool
        ``value`` as a plain ``bool``.

    Raises
    ------
    TypeError
        If ``value`` is not a bool (an int such as 1, or a str such as
        ``"yes"``): ``"<name> must be a bool, got <type>"``.

    Examples
    --------
    In the ``__post_init__`` of a frozen dataclass:

    >>> object.__setattr__(self, "flag", boolean("flag", self.flag))  # doctest: +SKIP
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def points(x: object, dim: int) -> np.ndarray:
    """Check an array of points and return it as floats.

    Parameters
    ----------
    x : object
        Points, array-like of shape ``(..., dim)``.
    dim : int
        The space dimension.

    Returns
    -------
    numpy.ndarray
        ``x`` as a float array.

    Raises
    ------
    ValueError
        If the last axis of ``x`` is not of length ``dim``:
        ``"points must have shape (..., <dim>), got <shape>"``.
    """
    out = np.asarray(x, dtype=float)
    if out.ndim == 0 or out.shape[-1] != dim:
        raise ValueError(f"points must have shape (..., {dim}), got {out.shape}")
    return out
