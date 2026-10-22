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


def test_records_are_deep_copies_in_plain_dicts():
    hparams = {"class_path": "m.S", "init_args": {"n": 8}}
    info = {"time": 1.0}
    result = Result(SQUARE, _u, info=info, hparams=hparams)
    info["time"] = 2.0
    hparams["init_args"]["n"] = 5
    assert result.info == {"time": 1.0}
    assert result.hparams["init_args"]["n"] == 8
    assert type(result.info) is dict and type(result.hparams) is dict


def test_node_arrays_are_read_only_copies():
    values = np.zeros(2)
    result = Result(SQUARE, _u, nodes=np.zeros((2, 2)), values=values)
    values[0] = 1.0
    assert result.values[0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.values[0] = 1.0


def test_optional_fields_are_keyword_only():
    with pytest.raises(TypeError):
        Result(SQUARE, _u, None)
