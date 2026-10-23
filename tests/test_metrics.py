"""The shared evaluation protocol: evaluate and summarize (ADR 0007)."""

import numpy as np
import pytest

from maesolvers import T1, T2, T8, T12, Box, Dirichlet, Disk, Problem, Result
from maesolvers.metrics import (
    Summary,
    _hessian_fd,
    _quantile,
    evaluate,
    seed_error,
    shared_grid,
    summarize,
    summarize_errors,
)


def _shifted(problem, shift):
    return lambda x: problem.exact(x) + shift


def _exact_hessian_t8(x):
    """Hessian of |x|^2 / 2: the identity."""
    return np.broadcast_to(np.eye(2), (*np.shape(x)[:-1], 2, 2))


# shared grid


def test_shared_grid_of_a_box_keeps_its_boundary():
    grid = shared_grid(T1.domain)
    assert grid.shape == (201 * 201, 2)
    assert np.any(np.all(grid == (1.0, 1.0), axis=-1))


def test_shared_grid_of_a_disk_drops_its_boundary():
    grid = shared_grid(T8.domain)
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
    m = evaluate(Result(T1, _shifted(T1, 1e-3)))
    assert m["max"] == pytest.approx(1e-3) and m["rms"] == pytest.approx(1e-3)
    assert "max_grid" not in m and "nonconvex_fraction" not in m


def test_grid_result_is_measured_at_its_nodes_and_on_the_grid():
    nodes = np.array([[0.5, 0.5], [0.25, 0.75]])
    result = Result(
        T1,
        _shifted(T1, 2e-3),
        nodes=nodes,
        values=T1.exact(nodes) + np.array([1e-3, -3e-3]),
    )
    m = evaluate(result)
    assert m["max"] == pytest.approx(3e-3)
    assert m["rms"] == pytest.approx(np.sqrt(5e-6))
    assert m["max_grid"] == pytest.approx(2e-3) and m["rms_grid"] == pytest.approx(2e-3)


def test_transport_errors_ignore_the_constant():
    nodes = shared_grid(T12.domain)[::50]
    result = Result(T12, _shifted(T12, 7.0), nodes=nodes, values=T12.exact(nodes) - 4.0)
    m = evaluate(result)
    for key in ("max", "rms", "max_grid", "rms_grid"):
        assert m[key] == pytest.approx(0.0, abs=1e-12)


def test_dirichlet_errors_see_the_constant():
    assert evaluate(Result(T1, _shifted(T1, 7.0)))["max"] == pytest.approx(7.0)


def test_nan_solution_gives_nan_errors():
    m = evaluate(Result(T1, lambda x: np.full(np.shape(x)[:-1], np.nan)))
    assert np.isnan(m["max"]) and np.isnan(m["rms"])


def _node_result(nodes, hess):
    nodes = np.asarray(nodes, dtype=float)
    return Result(T1, T1.exact, nodes=nodes, values=T1.exact(nodes), node_hessian=hess)


def test_node_hessian_convexity_ignores_boundary_nodes():
    # (0, 0) and (1, 0.5) lie on the boundary of T1's unit square: their
    # Hessians, NaN and finite non-convex, must not count.
    nodes = [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8], [0.0, 0.0], [1.0, 0.5]]
    hess = np.array(
        [
            np.eye(2),
            np.diag([2.0, -1.0]),
            np.diag([3.0, 0.5]),
            np.full((2, 2), np.nan),
            np.diag([-5.0, 1.0]),
        ]
    )
    m = evaluate(_node_result(nodes, hess))
    assert m["nonconvex_fraction"] == pytest.approx(1 / 3)
    assert m["eigmin"] == -1.0
    # Hessian of exp(|x|^2/2) at (0.2, 0.2): smallest eigenvalue exp(0.04).
    assert m["eigmin_exact"] == pytest.approx(np.exp(0.04), rel=1e-6)


def test_node_hessian_convexity_uses_the_method_interior_mask():
    # Nodes placed on the unit circle with cos/sin: the floating-point test
    # of Disk.interior counts some of them as interior, the method knows better.
    t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rim = np.stack([np.cos(t), np.sin(t)], axis=-1)
    assert np.any(T8.domain.interior(rim))
    inside = 0.5 * rim[::10]
    nodes = np.concatenate([rim, inside])
    hess = np.concatenate(
        [np.full((len(rim), 2, 2), np.nan), _exact_hessian_t8(inside)]
    )
    interior = np.arange(len(nodes)) >= len(rim)
    result = Result(
        T8,
        T8.exact,
        nodes=nodes,
        values=T8.exact(nodes),
        node_hessian=hess,
        interior=interior,
    )
    m = evaluate(result)
    assert m["nonconvex_fraction"] == 0.0 and m["eigmin"] == 1.0


def test_nan_node_hessian_at_an_interior_node_is_not_convex():
    nodes = [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]
    hess = np.array([np.eye(2), np.full((2, 2), np.nan), np.full((2, 2), np.nan)])
    m = evaluate(_node_result(nodes, hess))
    assert np.isnan(m["nonconvex_fraction"]) and np.isnan(m["eigmin"])


def test_nan_callable_hessian_on_the_shared_grid_is_not_convex():
    def hessian(x):
        out = _exact_hessian_t8(x).copy()
        out[1:] = np.nan
        return out

    m = evaluate(Result(T8, T8.exact, hessian=hessian))
    assert np.isnan(m["nonconvex_fraction"]) and np.isnan(m["eigmin"])
    assert m["eigmin_exact"] == pytest.approx(1.0, abs=1e-6)


def test_callable_hessian_convexity_on_the_shared_grid():
    m = evaluate(Result(T8, T8.exact, hessian=_exact_hessian_t8))
    assert m["nonconvex_fraction"] == 0.0
    assert m["eigmin"] == 1.0 and m["eigmin_exact"] == pytest.approx(1.0, abs=1e-6)


def test_concave_branch_is_fully_nonconvex():
    concave = Result(
        T8, lambda x: -T8.exact(x), hessian=lambda x: -_exact_hessian_t8(x)
    )
    m = evaluate(concave)
    assert m["nonconvex_fraction"] == 1.0 and m["eigmin"] == -1.0


def test_exact_eigmin_ignores_points_where_the_stencil_leaves_the_domain():
    # T2's exact solution is NaN beyond the corner (1, 1) of the shared grid.
    hess = lambda x: _hessian_fd(T2.exact, np.asarray(x) - 1e-3)  # noqa: E731
    m = evaluate(Result(T2, T2.exact, hessian=hess))
    assert np.isfinite(m["eigmin_exact"]) and m["eigmin_exact"] > 0


def test_cost_fields_come_from_info():
    info = {"time": 0.5, "iterations": 4, "unknowns": 900, "converged": True}
    m = evaluate(Result(T1, T1.exact, info=info))
    assert (m["time"], m["iterations"], m["unknowns"]) == (0.5, 4, 900)
    assert "converged" not in m


def test_observables_are_reported_by_name():
    problem = Problem(
        "U0",
        Disk((0, 0), 1),
        f=T8.f,
        boundary=Dirichlet(T8.boundary.g),
        observables={"u_origin": lambda u: u(np.zeros(2))},
    )
    m = evaluate(Result(problem, lambda x: 0.75 + T8.exact(x)))
    assert m == {"u_origin": 0.25}


def test_observable_may_not_shadow_a_metric():
    problem = Problem(
        "BAD", T1.domain, T1.f, T1.boundary, observables={"max": lambda u: 0.0}
    )
    with pytest.raises(ValueError, match="shadows"):
        evaluate(Result(problem, T1.exact))


def test_hessian_fd_is_second_order_accurate():
    points = np.array([[0.3, 0.7], [0.5, 0.1]])
    hess = _hessian_fd(T1.exact, points)
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


def _seed(error):
    """A result on T1 whose max error on the shared grid is ``error``."""
    return Result(T1, _shifted(T1, error))


def test_summarize_all_seeds_succeed():
    results = [_seed(e) for e in np.linspace(1e-3, 1e-2, 10)]
    s = summarize(results, threshold=5.5e-3)
    assert s == Summary(
        median=pytest.approx(5.5e-3),
        q05=pytest.approx(1.45e-3),
        q95=pytest.approx(9.55e-3),
        success_rate=0.5,
        seeds=10,
        failed=0,
    )


def test_summarize_counts_failed_seeds_as_infinite():
    results = [_seed(1e-3)] * 6 + [_seed(np.nan), _seed(np.inf), None, None]
    s = summarize(results, threshold=1e-2)
    assert s.failed == 4 and s.success_rate == 0.6
    assert s.median == pytest.approx(1e-3) and s.q95 == np.inf
    assert not np.isnan([s.median, s.q05, s.q95]).any()


def test_summarize_errors_applies_the_same_failed_seed_rule():
    errors = [1e-3] * 6 + [np.nan, np.inf, None, None]
    s = summarize_errors(errors, threshold=np.inf)
    assert s.failed == 4 and s.success_rate == 0.6 and s.q95 == np.inf
    results = [_seed(1e-3)] * 6 + [_seed(np.nan), _seed(np.inf), None, None]
    measured = [None if r is None else evaluate(r)["max"] for r in results]
    assert summarize(results, threshold=1e-2) == summarize_errors(measured, 1e-2)
    with pytest.raises(ValueError, match="at least one seed"):
        summarize_errors([], threshold=1.0)
    with pytest.raises(ValueError, match="threshold must be positive"):
        summarize_errors([1.0], threshold=0.0)


def test_summarize_keeps_an_unconverged_seed_with_a_finite_error():
    """Stopping at the step budget is not failing (the rule settled in #12)."""
    unconverged = Result(
        T1, _shifted(T1, 1e-3), info={"converged": False, "stop": "max_steps"}
    )
    s = summarize([unconverged, _seed(1e-1)], threshold=1e-2)
    assert s.failed == 0 and s.success_rate == 0.5


def test_summarize_median_fails_when_more_than_half_fail():
    results = [_seed(1e-3)] * 4 + [None] * 6
    s = summarize(results, threshold=1e-2)
    assert s.median == np.inf and s.q05 == pytest.approx(1e-3)


def test_summarize_median_fails_when_exactly_half_fail():
    assert summarize([_seed(1e-3)] * 5 + [None] * 5, threshold=1e-2).median == np.inf


def test_summarize_reads_the_requested_key():
    info = {"time": 0.1}
    s = summarize([Result(T1, T1.exact, info=info)], threshold=0.5, key="time")
    assert s.median == 0.1 and s.success_rate == 1.0


def test_summarize_needs_the_key_in_every_result():
    no_exact = Problem("NOEXACT", T1.domain, T1.f, T1.boundary)
    with pytest.raises(KeyError):
        summarize([Result(no_exact, T1.exact)], threshold=1.0)


@pytest.mark.parametrize(("seeds", "threshold"), [(0, 1.0), (1, 0.0)])
def test_summarize_rejects_bad_input(seeds, threshold):
    with pytest.raises(ValueError):
        summarize([_seed(1.0)] * seeds, threshold)


def _diverged(error: float) -> Result:
    """A neural seed that stopped on a non-finite loss, its network's error finite."""
    return Result(
        T1, _shifted(T1, error), info={"converged": False, "stop": "diverged"}
    )


def test_summarize_counts_a_diverged_seed_as_failed():
    """A run that stopped on a non-finite loss fails, its finite error aside (#14)."""
    results = [_diverged(1e-3), _seed(1e-3), _seed(1e-3)]
    s = summarize(results, threshold=1e-2)
    assert s.failed == 1 and s.success_rate == pytest.approx(2 / 3)
    assert s.q95 == np.inf and s.median == pytest.approx(1e-3)
    assert s == summarize_errors([seed_error(r) for r in results], threshold=1e-2)


def test_seed_error_is_none_for_a_failed_seed_only():
    assert seed_error(_seed(1e-3)) == pytest.approx(1e-3)
    assert seed_error(_seed(1e-3), key="rms") == evaluate(_seed(1e-3))["rms"]
    for failed in (None, _diverged(1e-3), _seed(np.nan), _seed(np.inf)):
        assert seed_error(failed) is None
    budget = Result(T1, _shifted(T1, 1e-3), info={"stop": "max_steps"})
    assert seed_error(budget) == pytest.approx(1e-3)
