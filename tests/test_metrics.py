"""The shared evaluation protocol: evaluate and summarize (ADR 0007)."""

import numpy as np
import pytest

from maesolvers import P1, P3, P4, P5, Box, Dirichlet, Disk, Problem, Result
from maesolvers.metrics import (
    Summary,
    _hessian_fd,
    _quantile,
    evaluate,
    shared_grid,
    summarize,
)


def _shifted(problem, shift):
    return lambda x: problem.exact(x) + shift


def _exact_hessian_p1(x):
    """Hessian of |x|^2 / 2: the identity."""
    return np.broadcast_to(np.eye(2), (*np.shape(x)[:-1], 2, 2))


# shared grid


def test_shared_grid_of_a_box_keeps_its_boundary():
    grid = shared_grid(P4.domain)
    assert grid.shape == (201 * 201, 2)
    assert np.any(np.all(grid == (1.0, 1.0), axis=-1))


def test_shared_grid_of_a_disk_drops_its_boundary():
    grid = shared_grid(P1.domain)
    assert np.all(np.linalg.norm(grid, axis=-1) < 1.0)
    # linspace(-1, 1, 201) is k / 100: count the integer points with i^2 + j^2 < 100^2.
    k = np.arange(-100, 101)
    assert len(grid) == np.sum(k[:, None] ** 2 + k[None, :] ** 2 < 100**2)


@pytest.mark.parametrize(("dim", "size"), [(3, 51), (4, 21)])
def test_shared_grid_size_follows_the_dimension(dim, size):
    assert shared_grid(Box((0,) * dim, (1,) * dim)).shape == (size**dim, dim)


def test_shared_grid_size_can_be_given():
    assert shared_grid(Box((0,) * 5, (1,) * 5), size=3).shape == (3**5, 5)
    with pytest.raises(ValueError, match="dimension 5"):
        shared_grid(Box((0,) * 5, (1,) * 5))


# evaluate


def test_neural_style_result_is_measured_on_the_shared_grid():
    m = evaluate(Result(P4, _shifted(P4, 1e-3)))
    assert m["max"] == pytest.approx(1e-3) and m["rms"] == pytest.approx(1e-3)
    assert "max_grid" not in m and "nonconvex_fraction" not in m


def test_grid_result_is_measured_at_its_nodes_and_on_the_grid():
    nodes = np.array([[0.5, 0.5], [0.25, 0.75]])
    result = Result(
        P4, _shifted(P4, 2e-3), nodes=nodes, values=P4.exact(nodes) + [1e-3, -3e-3]
    )
    m = evaluate(result)
    assert m["max"] == pytest.approx(3e-3)
    assert m["rms"] == pytest.approx(np.sqrt(5e-6))
    assert m["max_grid"] == pytest.approx(2e-3) and m["rms_grid"] == pytest.approx(2e-3)


def test_transport_errors_ignore_the_constant():
    nodes = shared_grid(P3.domain)[::50]
    result = Result(P3, _shifted(P3, 7.0), nodes=nodes, values=P3.exact(nodes) - 4.0)
    m = evaluate(result)
    for key in ("max", "rms", "max_grid", "rms_grid"):
        assert m[key] == pytest.approx(0.0, abs=1e-12)


def test_dirichlet_errors_see_the_constant():
    assert evaluate(Result(P4, _shifted(P4, 7.0)))["max"] == pytest.approx(7.0)


def test_nan_solution_gives_nan_errors():
    m = evaluate(Result(P4, lambda x: np.full(np.shape(x)[:-1], np.nan)))
    assert np.isnan(m["max"]) and np.isnan(m["rms"])


def test_node_hessian_convexity_skips_undefined_nodes():
    nodes = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8], [0.0, 0.0]])
    hess = np.array(
        [np.eye(2), np.diag([2.0, -1.0]), np.diag([3.0, 0.5]), np.full((2, 2), np.nan)]
    )
    result = Result(
        P4, P4.exact, nodes=nodes, values=P4.exact(nodes), node_hessian=hess
    )
    m = evaluate(result)
    assert m["nonconvex_fraction"] == pytest.approx(1 / 3)
    assert m["eigmin"] == -1.0
    # Hessian of exp(|x|^2/2) at (0.2, 0.2): smallest eigenvalue exp(0.04).
    assert m["eigmin_exact"] == pytest.approx(np.exp(0.04), rel=1e-6)


def test_callable_hessian_convexity_on_the_shared_grid():
    m = evaluate(Result(P1, P1.exact, hessian=_exact_hessian_p1))
    assert m["nonconvex_fraction"] == 0.0
    assert m["eigmin"] == 1.0 and m["eigmin_exact"] == pytest.approx(1.0, abs=1e-6)


def test_concave_branch_is_fully_nonconvex():
    concave = Result(
        P1, lambda x: 1.0 - P1.exact(x), hessian=lambda x: -_exact_hessian_p1(x)
    )
    m = evaluate(concave)
    assert m["nonconvex_fraction"] == 1.0 and m["eigmin"] == -1.0


def test_exact_eigmin_ignores_points_where_the_stencil_leaves_the_domain():
    # P5's exact solution is NaN beyond the corner (1, 1) of the shared grid.
    hess = lambda x: _hessian_fd(P5.exact, np.asarray(x) - 1e-3)  # noqa: E731
    m = evaluate(Result(P5, P5.exact, hessian=hess))
    assert np.isfinite(m["eigmin_exact"]) and m["eigmin_exact"] > 0


def test_cost_fields_come_from_info():
    info = {"time": 0.5, "iterations": 4, "unknowns": 900, "converged": True}
    m = evaluate(Result(P4, P4.exact, info=info))
    assert (m["time"], m["iterations"], m["unknowns"]) == (0.5, 4, 900)
    assert "converged" not in m


def test_observables_are_reported_by_name():
    problem = Problem(
        "U0",
        Disk((0, 0), 1),
        f=P1.f,
        boundary=Dirichlet(P1.boundary.g),
        observables={"u(0,0)": lambda u: u(np.zeros(2))},
    )
    m = evaluate(Result(problem, lambda x: 0.25 + P1.exact(x)))
    assert m == {"u(0,0)": 0.25}


def test_observable_may_not_shadow_a_metric():
    problem = Problem(
        "BAD", P4.domain, P4.f, P4.boundary, observables={"max": lambda u: 0.0}
    )
    with pytest.raises(ValueError, match="shadows"):
        evaluate(Result(problem, P4.exact))


def test_hessian_fd_is_second_order_accurate():
    points = np.array([[0.3, 0.7], [0.5, 0.1]])
    hess = _hessian_fd(P4.exact, points)
    r2 = np.sum(points**2, axis=-1)
    expected = np.exp(r2 / 2)[:, None, None] * (
        np.eye(2) + points[:, :, None] * points[:, None, :]
    )
    assert np.allclose(hess, expected, rtol=1e-6)


# summarize


def test_quantile_matches_numpy_on_finite_values():
    values = np.sort(np.random.default_rng(0).uniform(size=10))
    for q in (0.05, 0.5, 0.95):
        assert _quantile(values, q) == pytest.approx(np.quantile(values, q))


def test_summarize_all_seeds_succeed():
    metrics = [{"max": e} for e in np.linspace(1e-3, 1e-2, 10)]
    s = summarize(metrics, threshold=5.5e-3)
    assert s == Summary(
        median=pytest.approx(5.5e-3),
        q05=pytest.approx(1.45e-3),
        q95=pytest.approx(9.55e-3),
        success_rate=0.5,
        seeds=10,
        failed=0,
    )


def test_summarize_counts_failed_seeds_as_infinite():
    metrics = [{"max": 1e-3}] * 6 + [{"max": np.nan}, {"max": np.inf}, None, None]
    s = summarize(metrics, threshold=1e-2)
    assert s.failed == 4 and s.success_rate == 0.6
    assert s.median == pytest.approx(1e-3) and s.q95 == np.inf
    assert not np.isnan([s.median, s.q05, s.q95]).any()


def test_summarize_median_fails_when_more_than_half_fail():
    metrics = [{"max": 1e-3}] * 4 + [None] * 6
    s = summarize(metrics, threshold=1e-2)
    assert s.median == np.inf and s.q05 == pytest.approx(1e-3)


def test_summarize_median_fails_when_exactly_half_fail():
    assert summarize([{"max": 1e-3}] * 5 + [None] * 5, threshold=1e-2).median == np.inf


def test_summarize_reads_the_requested_key():
    s = summarize([{"max": 1.0, "rms": 0.1}], threshold=0.5, key="rms")
    assert s.median == 0.1 and s.success_rate == 1.0


@pytest.mark.parametrize(
    ("metrics", "threshold", "error"),
    [
        ([], 1.0, ValueError),
        ([{"max": 1.0}], 0.0, ValueError),
        ([{"rms": 1.0}], 1.0, KeyError),
    ],
)
def test_summarize_rejects_bad_input(metrics, threshold, error):
    with pytest.raises(error):
        summarize(metrics, threshold)
