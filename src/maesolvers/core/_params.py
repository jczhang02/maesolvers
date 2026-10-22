"""Conversion of constructor parameters, with errors that name the parameter."""

from __future__ import annotations

import numbers
import operator


def integer(name: str, value: object, minimum: int) -> int:
    """``value`` as a plain int of at least ``minimum``.

    Accepts Python and numpy integers; rejects bool, float and str.
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
    """``value`` as a plain finite float greater than zero.

    Accepts Python and numpy real numbers; rejects bool and str.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not 0.0 < out < float("inf"):
        raise ValueError(f"{name} must be positive and finite, got {out}")
    return out
