import numpy as np
import pytest

from maesolvers import (
    T1,
    T2,
    T8,
    Box,
    Dirichlet,
    FDNewton,
    Problem,
    UnsupportedProblemError,
)
from maesolvers.metrics import evaluate


def test_second_order_convergence():
    errors = []
    for n in (16, 32, 64):
        result = FDNewton(n=n).solve(T1)
        assert result.info["converged"]
        errors.append(evaluate(result)["max"])
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 1.7), rates


def test_solution_is_convex_on_t1():
    result = FDNewton(n=32).solve(T1)
    assert evaluate(result)["nonconvex_fraction"] == 0.0
    assert result.info["residual_max"] < 1e-8


def test_corner_blowup_breaks_the_standard_scheme():
    """Characterisation, not a goal: the non-monotone scheme converges in the
    residual but to a discrete solution that is non-convex near the corner,
    and the max error does not decrease with the grid (see benchmarks/)."""
    results = {n: FDNewton(n=n).solve(T2) for n in (32, 64)}
    assert all(r.info["converged"] for r in results.values())
    metrics = {n: evaluate(r) for n, r in results.items()}
    assert metrics[64]["nonconvex_fraction"] > 0.0
    assert metrics[64]["max"] > 0.5 * metrics[32]["max"]


def test_disk_domain_is_rejected():
    assert not FDNewton().supports(T8)
    with pytest.raises(UnsupportedProblemError, match="Disk"):
        FDNewton().solve(T8)


def test_a_start_that_meets_tol_is_converged():
    """For u = |x|^2 / 2 the scheme is exact, so the Poisson start is the solution."""
    half_r2 = lambda x: 0.5 * np.sum(np.asarray(x) ** 2, axis=-1)  # noqa: E731
    one = lambda x: np.ones(np.shape(x)[:-1])  # noqa: E731
    problem = Problem("Q", Box((0, 0), (1, 1)), one, Dirichlet(half_r2), exact=half_r2)
    result = FDNewton(n=8, tol=1e-8).solve(problem)
    assert result.info["converged"] and result.info["iterations"] == 1
    assert evaluate(result)["max"] < 1e-12
