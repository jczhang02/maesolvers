"""Result: the shapes it accepts and what it freezes."""

import numpy as np
import pytest

from maesolvers.core import Box, Dirichlet, Problem, Result


def _u(x):
    return np.zeros(np.shape(x)[:-1])


SQUARE = Problem("SQUARE", Box((0, 0), (1, 1)), _u, Dirichlet(_u))


def test_a_callable_alone_is_a_result():
    result = Result(SQUARE, _u)
    assert result.nodes is None and result.hessian is None
    assert dict(result.info) == {} and dict(result.hparams) == {}


def test_node_arrays_are_converted_to_float():
    result = Result(SQUARE, _u, nodes=[[0, 0], [1, 1]], values=[0, 1])
    assert result.nodes.dtype == float and result.values.shape == (2,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(nodes=np.zeros((3, 2))), "go together"),
        (dict(values=np.zeros(3)), "go together"),
        (dict(nodes=np.zeros((3, 3)), values=np.zeros(3)), r"\(M, 2\)"),
        (dict(nodes=np.zeros((3, 2)), values=np.zeros(4)), r"\(3,\)"),
        (
            dict(
                nodes=np.zeros((3, 2)),
                values=np.zeros(3),
                node_hessian=np.zeros((3, 2)),
            ),
            r"\(3, 2, 2\)",
        ),
        (dict(node_hessian=np.zeros((3, 2, 2))), "needs Result.nodes"),
    ],
)
def test_inconsistent_node_arrays_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Result(SQUARE, _u, **kwargs)


@pytest.mark.parametrize("kwargs", [dict(u=1.0), dict(u=_u, hessian=1.0)])
def test_solution_and_hessian_must_be_callable(kwargs):
    with pytest.raises(TypeError, match="callable"):
        Result(SQUARE, **kwargs)


def test_info_is_a_read_only_copy():
    info = {"time": 1.0}
    result = Result(SQUARE, _u, info=info)
    info["time"] = 2.0
    assert result.info["time"] == 1.0
    with pytest.raises(TypeError):
        result.info["time"] = 3.0
