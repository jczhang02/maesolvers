"""Problem and boundary condition types."""

import numpy as np
import pytest

from maesolvers.core import Box, Dirichlet, Disk, Ellipse, Problem, Transport


def _f(x):
    return np.ones(np.shape(x)[:-1])


def _problem(**kwargs):
    defaults = dict(name="X", domain=Box((0, 0), (1, 1)), f=_f, boundary=Dirichlet(_f))
    return Problem(**{**defaults, **kwargs})


def test_dirichlet_needs_a_callable():
    with pytest.raises(TypeError, match="Dirichlet.g"):
        Dirichlet(1.0)


def test_transport_needs_a_domain_target():
    with pytest.raises(TypeError, match="Transport.target"):
        Transport((0.0, 0.0))


def test_problem_carries_its_dimension():
    assert _problem(domain=Box((0,) * 3, (1,) * 3)).dim == 3


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (dict(name=""), ValueError, "name"),
        (dict(domain=(0, 1)), TypeError, "domain"),
        (dict(f=2.0), TypeError, "Problem.f"),
        (dict(boundary="dirichlet"), TypeError, "Dirichlet or Transport"),
        (dict(exact=0.0), TypeError, "exact"),
        (dict(observables={"u0": 1.0}), TypeError, "u0"),
        (
            dict(
                domain=Box((0,) * 3, (1,) * 3),
                boundary=Transport(Ellipse((0, 0), (1, 1))),
            ),
            ValueError,
            "2D but the domain is 3D",
        ),
    ],
)
def test_problem_rejects_bad_parts(kwargs, error, message):
    with pytest.raises(error, match=message):
        _problem(**kwargs)


def test_observables_are_read_only():
    problem = _problem(observables={"u0": lambda u: float(u(np.zeros(2)))})
    with pytest.raises(TypeError):
        problem.observables["u1"] = lambda u: 0.0


def test_problems_compare_by_identity():
    a = _problem(domain=Disk((0, 0), 1), boundary=Transport(Ellipse((0, 0), (1, 1))))
    b = _problem(domain=Disk((0, 0), 1), boundary=Transport(Ellipse((0, 0), (1, 1))))
    assert a == a and a != b
    assert len({a, b}) == 2
