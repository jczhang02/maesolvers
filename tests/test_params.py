"""The parameter checks that solvers call in ``__post_init__``."""

import numpy as np
import pytest

from maesolvers.core import boolean, integer, nonnegative, positive


@pytest.mark.parametrize("value", [3, np.int64(3), np.int32(3)])
def test_integer_returns_a_plain_int(value):
    out = integer("n", value, 2)
    assert out == 3
    assert type(out) is int


@pytest.mark.parametrize("value", [True, 3.0, np.float64(3.0), "3", None])
def test_integer_rejects_a_non_integer_naming_the_parameter(value):
    with pytest.raises(TypeError, match="n must be an integer"):
        integer("n", value, 2)


def test_integer_rejects_a_value_below_the_minimum():
    assert integer("n", 2, 2) == 2
    with pytest.raises(ValueError, match="n must be at least 2, got 1"):
        integer("n", 1, 2)


@pytest.mark.parametrize("value", [1e-3, 1, np.float32(0.5), np.int64(2)])
def test_positive_returns_a_plain_float(value):
    out = positive("tol", value)
    assert out == float(value)
    assert type(out) is float


@pytest.mark.parametrize("value", [True, "1e-3", None, 1j])
def test_positive_rejects_a_non_real_naming_the_parameter(value):
    with pytest.raises(TypeError, match="tol must be a real number"):
        positive("tol", value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_positive_rejects_zero_negative_and_non_finite_values(value):
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        positive("tol", value)


@pytest.mark.parametrize("value", [0, 0.0, 1e4, np.float32(0.5), np.int64(2)])
def test_nonnegative_accepts_zero_and_returns_a_plain_float(value):
    out = nonnegative("weight", value)
    assert out == float(value)
    assert type(out) is float


@pytest.mark.parametrize("value", [False, "0", None])
def test_nonnegative_rejects_a_non_real_naming_the_parameter(value):
    with pytest.raises(TypeError, match="weight must be a real number"):
        nonnegative("weight", value)


@pytest.mark.parametrize("value", [-1e-12, float("inf"), float("nan")])
def test_nonnegative_rejects_negative_and_non_finite_values(value):
    with pytest.raises(ValueError, match="weight must be nonnegative and finite"):
        nonnegative("weight", value)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_boolean_returns_a_plain_bool(value):
    out = boolean("flag", value)
    assert out == bool(value)
    assert type(out) is bool


@pytest.mark.parametrize("value", [1, 0, "yes", None, np.int64(1)])
def test_boolean_rejects_a_non_bool_naming_the_parameter(value):
    with pytest.raises(TypeError, match="flag must be a bool"):
        boolean("flag", value)
