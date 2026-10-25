"""The hybrid scheme, its switching rule and its preset."""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from maesolvers import (
    SOLVERS,
    T1,
    T2,
    T8,
    T20,
    Box,
    Dirichlet,
    FDHybrid,
    FDMonotone,
    FDNewton,
    Hybrid,
    Monotone17,
    Newton,
    PoissonIteration,
    Problem,
    Standard9,
    UnsupportedProblemError,
    from_config,
)
from maesolvers.classical.hybrid import hybrid_weights
from maesolvers.core.callbacks import Callback
from maesolvers.metrics import evaluate

UNIT = Box((0.0, 0.0), (1.0, 1.0))


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


def _bowl(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.sum(x**2, axis=-1) + 0.1 * x[..., 0] * x[..., 1]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Everywhere(Hybrid):
    """A switching rule of one's own: the same weight at every unknown."""

    w: float = 1.0

    def weights(self, problem, nodes, h):
        return np.full(len(nodes), self.w)


def _weights(scheme, problem):
    """The weights a scheme fixes for a problem."""
    return scheme.discretize(problem).weight


def test_the_rule_puts_the_boundary_ring_in_the_monotone_band():
    w = hybrid_weights(np.ones((7, 7)), epsilon=0.1, band=1.0)
    ring = np.ones((7, 7), dtype=bool)
    ring[1:-1, 1:-1] = False
    assert np.all(w[ring] == 0.0) and np.all(w[~ring] == 1.0)


@pytest.mark.parametrize("value", [0.05, 0.1, 10.0, 20.0, np.inf, np.nan])
def test_the_rule_marks_f_at_most_epsilon_or_at_least_its_inverse(value):
    f = np.ones((9, 9))
    f[4, 4] = value
    w = hybrid_weights(f, epsilon=0.1, band=1.0)
    # the node and its four axis neighbours, one step away; not the diagonal ones
    assert w[4, 4] == w[3, 4] == w[5, 4] == w[4, 3] == w[4, 5] == 0.0
    assert w[3, 3] == w[2, 4] == 1.0


def test_the_rule_keeps_the_regular_values_between_the_thresholds():
    f = np.full((9, 9), 0.11)
    f[4, 4] = 9.9
    assert hybrid_weights(f, epsilon=0.1, band=1.0)[1:-1, 1:-1].min() == 1.0


def test_the_band_is_a_euclidean_radius_in_grid_steps():
    f = np.ones((11, 11))
    f[5, 5] = 0.0
    zero = {
        band: int(np.sum(hybrid_weights(f, 0.1, band)[2:-2, 2:-2] == 0.0))
        for band in (0.0, 1.0, np.sqrt(2), 2.0)
    }
    assert zero == {0.0: 1, 1.0: 5, np.sqrt(2): 9, 2.0: 13}
    assert np.all(hybrid_weights(np.ones((5, 5)), 0.1, 0.5) == 1.0)


def test_t1_is_regular_away_from_the_boundary_and_t2_singular_at_the_corner():
    n = 32
    t1 = _weights(Hybrid(n=n), T1).reshape(n - 1, n - 1)
    assert np.all(t1[1:-1, 1:-1] == 1.0)
    t2 = _weights(Hybrid(n=n), T2).reshape(n - 1, n - 1)
    assert t2[-2, -2] == 0.0 and t2[1, 1] == 1.0  # f >= 1/h next to (1, 1)


def test_t20_uses_the_monotone_scheme_on_its_flat_disk():
    n = 40
    grid = Hybrid(n=n).discretize(T20)
    r = np.linalg.norm(grid.monotone.nodes[grid.monotone.interior] - 0.5, axis=-1)
    w = grid.weight
    assert np.all(w[r <= 0.2] == 0.0) and np.all(w[(r > 0.3) & (r < 0.4)] == 1.0)


def test_the_weights_follow_the_order_of_the_unknowns():
    """An off-diagonal node where f = 0, and its mirror image, tell x from y."""
    n, (i, j) = 8, (2, 5)
    node = np.array([i / n, j / n])

    def f(x):
        return np.where(np.all(np.isclose(x, node), axis=-1), 0.0, 1.0)

    problem = Problem("asymmetric", UNIT, f, Dirichlet(_bowl))
    scheme = Hybrid(n=n, band=0.0)
    grid = scheme.discretize(problem)
    k, mirror = (i - 1) * (n - 1) + j - 1, (j - 1) * (n - 1) + i - 1
    assert np.flatnonzero(grid.weight == 0.0).tolist() == [k]
    v = _bowl(grid.monotone.nodes[grid.monotone.interior])
    residual = scheme.residual(grid, v)
    assert residual[k] == scheme.monotone.residual(grid.monotone, v)[k]
    assert residual[mirror] == scheme.standard.residual(grid.standard, v)[mirror]


def test_a_rule_of_ones_own_gets_the_nodes_of_the_unknowns():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class MonotoneOnTheLeft(Hybrid):
        def weights(self, problem, nodes, h):
            assert h == pytest.approx(1 / self.n)
            return np.where(nodes[:, 0] < 0.5, 0.0, 1.0)

    n = 8
    scheme = MonotoneOnTheLeft(n=n)
    grid = scheme.discretize(T1)
    w = grid.weight.reshape(n - 1, n - 1)  # w[i - 1, j - 1] at node (i, j)
    assert np.all(w[: n // 2 - 1] == 0.0) and np.all(w[n // 2 - 1 :] == 1.0)
    v = _bowl(grid.monotone.nodes[grid.monotone.interior])
    left = slice(0, (n // 2 - 1) * (n - 1))
    np.testing.assert_array_equal(
        scheme.residual(grid, v)[left],
        scheme.monotone.residual(grid.monotone, v)[left],
    )


def test_epsilon_defaults_to_the_grid_step():
    a = _weights(Hybrid(n=20), T2)
    b = _weights(Hybrid(n=20, epsilon=1 / 20), T2)
    np.testing.assert_array_equal(a, b)
    assert np.sum(_weights(Hybrid(n=20, epsilon=0.3), T2) == 0) > np.sum(a == 0)


@pytest.mark.parametrize(("w", "other"), [(1.0, Standard9), (0.0, Monotone17)])
def test_a_constant_weight_is_one_of_the_two_schemes(w, other):
    scheme = _Everywhere(n=9, w=w)
    grid = scheme.discretize(T1)
    part = other(n=9)
    part_grid = part.discretize(T1)
    v = _bowl(grid.monotone.nodes[grid.monotone.interior])
    np.testing.assert_allclose(
        scheme.residual(grid, v), part.residual(part_grid, v), rtol=1e-12
    )
    np.testing.assert_allclose(
        scheme.jacobian(grid, v).toarray(),
        part.jacobian(part_grid, v).toarray(),
        rtol=1e-12,
    )


def test_the_residual_and_jacobian_are_the_weighted_averages():
    scheme = Hybrid(n=9)
    grid = scheme.discretize(T2)
    v = _bowl(grid.monotone.nodes[grid.monotone.interior])
    w = grid.weight
    assert 0 < np.mean(w) < 1
    expected = w * scheme.standard.residual(grid.standard, v) + (
        1 - w
    ) * scheme.monotone.residual(grid.monotone, v)
    np.testing.assert_array_equal(scheme.residual(grid, v), expected)


def test_jacobian_matches_finite_differences_of_the_residual():
    scheme = Hybrid(n=8)
    grid = scheme.discretize(T1)
    rng = np.random.default_rng(3)
    v = _bowl(grid.monotone.nodes[grid.monotone.interior])
    v = v + 1e-4 * rng.standard_normal(v.size)
    jacobian = scheme.jacobian(grid, v)
    assert isinstance(jacobian, sp.csc_matrix)
    eps = 1e-7
    numeric = np.column_stack(
        [
            (scheme.residual(grid, v + eps * e) - scheme.residual(grid, v - eps * e))
            / (2 * eps)
            for e in np.eye(v.size)
        ]
    )
    np.testing.assert_allclose(jacobian.toarray(), numeric, rtol=1e-5, atol=1e-6)


def test_on_the_smooth_example_it_lies_between_the_two_schemes():
    n = 31
    result = FDHybrid(n=n).solve(T1)
    standard = evaluate(FDNewton(n=n).solve(T1))["max"]
    monotone = evaluate(FDMonotone(n=n).solve(T1))["max"]
    assert result.info["converged"]
    assert standard < evaluate(result)["max"] < monotone


def test_where_f_vanishes_or_blows_up_everywhere_it_is_the_monotone_scheme():
    """The cone's discrete Dirac mass (JCP 2011, Section 8.5): w = 0 at every node."""
    n = 12

    def f(x):
        r = np.linalg.norm(np.asarray(x) - 0.5, axis=-1)
        return np.where(r <= 0.5 / n * (1 + 1e-12), 4.0 * n**2, 0.0)

    def cone(x):
        return np.linalg.norm(np.asarray(x) - 0.5, axis=-1)

    problem = Problem("cone", UNIT, f, Dirichlet(cone), exact=cone)
    assert np.all(_weights(Hybrid(n=n), problem) == 0.0)
    hybrid = FDHybrid(n=n).solve(problem)
    monotone = FDMonotone(n=n).solve(problem)
    np.testing.assert_allclose(hybrid.values, monotone.values, rtol=0, atol=1e-12)


def test_solution_reports_the_hybrid_residual_and_the_monotone_share():
    n = 16
    result = FDHybrid(n=n).solve(T1)
    assert result.info["residual_max"] <= 1e-10 * float(np.max(T1.f(result.nodes)))
    # the boundary ring alone: 4 (n - 2) of (n - 1)^2 unknowns
    assert result.info["monotone_fraction"] == pytest.approx(4 * (n - 2) / (n - 1) ** 2)
    assert result.info["interpolant"] == "spline(3)"


def test_supports_a_square_dirichlet_box_only():
    scheme = Hybrid()
    assert scheme.supports(T1) and not scheme.supports(T8)
    oblong = Problem("R", Box((0.0, 0.0), (2.0, 1.0)), _constant(1.0), Dirichlet(_bowl))
    assert not scheme.supports(oblong)
    with pytest.raises(UnsupportedProblemError, match="Hybrid does not support R"):
        scheme.discretize(oblong)
    with pytest.raises(
        UnsupportedProblemError,
        match=r"FDHybrid .*: Hybrid needs a Box with equal sides",
    ):
        FDHybrid(n=4).solve(oblong)
    with pytest.raises(UnsupportedProblemError, match="FDHybrid does not support T8"):
        FDHybrid(n=4).solve(T8)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (dict(n=1), ValueError, "n must be at least 2"),
        (dict(stencil=25), ValueError, "stencil must be 9, 17 or 33 points"),
        (dict(boundary="wide"), ValueError, "boundary must be 'narrow'"),
        (dict(epsilon=0.0), ValueError, "epsilon must be positive"),
        (dict(epsilon=1.0), ValueError, "epsilon must be less than 1"),
        (dict(epsilon="h"), TypeError, "epsilon must be a real number"),
        (dict(band=-1.0), ValueError, "band must be nonnegative"),
    ],
)
def test_hybrid_validates_its_parameters(kwargs, error, message):
    with pytest.raises(error, match=message):
        Hybrid(**kwargs)
    with pytest.raises(error, match=message):
        FDHybrid(**kwargs)


@pytest.mark.parametrize(
    ("w", "message"),
    [(np.ones(3), r"shape \(49,\)"), (np.full(49, 1.5), r"in \[0, 1\]")],
)
def test_a_rule_of_ones_own_must_return_one_weight_in_0_1_per_unknown(w, message):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Wrong(Hybrid):
        def weights(self, problem, nodes, h):
            return w

    with pytest.raises(ValueError, match=message):
        Wrong(n=8).discretize(T1)


def test_a_rule_of_ones_own_plugs_into_newton():
    half = Newton(scheme=_Everywhere(n=12, w=0.5)).solve(T1)
    assert half.info["converged"] and half.info["monotone_fraction"] == 0.5


def test_hybrid_is_a_frozen_keyword_only_part_built_from_the_two_schemes():
    scheme = Hybrid(n=np.int64(12), stencil=np.int64(33), boundary="interpolate")
    assert scheme == Hybrid(n=12, stencil=33, boundary="interpolate")
    assert scheme.standard == Standard9(n=12)
    assert scheme.monotone == Monotone17(n=12, stencil=33, boundary="interpolate")
    with pytest.raises(TypeError):
        Hybrid(8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Hybrid().band = 2.0


@pytest.mark.parametrize("problem", [T1, T2])
def test_the_preset_is_its_composition(problem):
    composed = Newton(
        scheme=Hybrid(n=12, band=2.0), start=FDMonotone(n=12, stencil=17), max_iter=200
    )
    ours = FDHybrid(n=12, band=2.0)
    assert isinstance(ours, Newton)
    assert (ours.scheme, ours.start) == (composed.scheme, composed.start)
    a, b = ours.solve(problem), composed.solve(problem)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.info["history"] == b.info["history"]


def test_newton_from_the_poisson_start_reaches_a_non_convex_root_on_t2():
    """Why the preset starts from the monotone root (benchmarks/hybrid.md)."""
    n = 63
    poisson = evaluate(Newton(scheme=Hybrid(n=n), max_iter=200).solve(T2))
    ours = evaluate(FDHybrid(n=n).solve(T2))
    # Which non-convex root Newton reaches depends on rounding
    # (standard_fd.md, Finding 4), so compare with the convex one.
    assert poisson["max"] > 3 * ours["max"]
    assert poisson["nonconvex_fraction"] > 2 * ours["nonconvex_fraction"]
    assert ours["max"] == pytest.approx(5.7213e-4, rel=1e-4)


def test_newton_starts_from_the_final_iterate_of_its_start():
    """The start's root is a root of the same scheme, so Newton takes no step."""
    start = FDMonotone(n=16)
    first = start.solve(T1)
    result = Newton(scheme=Monotone17(n=16), start=start).solve(T1)
    np.testing.assert_array_equal(result.values, first.values)
    assert result.info["iterations"] == 1 and result.info["history"][0] <= 1e-9
    assert result.info["start_iterations"] == first.info["iterations"]
    assert result.info["start_converged"] is True
    assert "start_iterations" not in FDNewton(n=8).solve(T1).info


def test_the_start_has_its_own_budget():
    result = FDHybrid(n=16, max_iter=1).solve(T2)
    assert result.info["iterations"] == 1
    assert result.info["start_converged"] and result.info["start_iterations"] > 1


def test_an_unconverged_start_warns_and_shows_in_info():
    solver = Newton(scheme=Hybrid(n=16), start=FDMonotone(n=16, max_iter=1))
    with pytest.warns(RuntimeWarning, match="Newton.start did not converge on T2"):
        result = solver.solve(T2)
    assert result.info["start_iterations"] == 1
    assert result.info["start_converged"] is False


def test_the_callbacks_see_the_steps_on_the_scheme_only():
    class Steps(Callback):
        def __init__(self):
            self.steps = []

        def on_step_end(self, solver, step, metrics):
            self.steps.append(step)

    steps = Steps()
    result = FDHybrid(n=16).solve(T1, callbacks=[steps])
    assert steps.steps == list(range(1, result.info["iterations"] + 1))
    assert result.info["start_iterations"] > result.info["iterations"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FromTheExactSolution(Newton):
    """A user's Newton with a start of its own."""

    def initial_guess(self, problem, context):
        return problem.exact(context.nodes[context.interior])


@pytest.mark.filterwarnings("ignore:Newton.start did not converge")
def test_any_scheme_iteration_is_a_start():
    """The paper's start (JCP Section 7.1) is one Poisson step; a subclass runs as is."""
    poisson = PoissonIteration(scheme=Standard9(n=16), max_iter=1)
    result = Newton(scheme=Hybrid(n=16), start=poisson).solve(T1)
    assert result.info["converged"] and result.info["start_iterations"] == 1
    assert result.info["start_converged"] is False
    exact = _FromTheExactSolution(scheme=Monotone17(n=16), max_iter=200)
    result = Newton(scheme=Hybrid(n=16), start=exact).solve(T1)
    assert result.info["start_iterations"] == exact.solve(T1).info["iterations"]


@pytest.mark.filterwarnings("ignore:Newton.start did not converge")
def test_newton_checks_its_start():
    with pytest.raises(TypeError, match=r"Newton\.start must be an instance"):
        Newton(scheme=Hybrid(n=8), start=FDMonotone)
    with pytest.raises(TypeError, match=r"Newton\.start must subclass SchemeIteration"):
        Newton(scheme=Hybrid(n=8), start=Monotone17(n=8))  # a scheme, not a solver
    with pytest.raises(ValueError, match="start has 81 unknowns on T1, the scheme 49"):
        Newton(scheme=Hybrid(n=8), start=FDMonotone(n=10, max_iter=1)).solve(T1)
    oblong = Problem("R", Box((0.0, 0.0), (2.0, 1.0)), _constant(1.0), Dirichlet(_bowl))
    mixed = Newton(scheme=Standard9(n=8), start=FDMonotone(n=8))
    assert mixed.supports(T1) and not mixed.supports(oblong)
    with pytest.raises(UnsupportedProblemError, match="Newton does not support R"):
        mixed.solve(oblong)


def test_the_preset_has_flat_hparams_and_its_scheme_defaults():
    assert FDHybrid().hparams == {
        "class_path": "maesolvers.FDHybrid",
        "init_args": {
            "n": 64,
            "stencil": 17,
            "boundary": "narrow",
            "jacobian_floor": 1e-12,
            "spline_degree": 3,
            "epsilon": None,
            "band": 1.0,
            "max_iter": 200,
            "tol": 1e-10,
        },
    }
    assert FDHybrid().scheme == Hybrid()
    assert SOLVERS["FDHybrid"] is FDHybrid


@pytest.mark.parametrize(
    "solver",
    [
        FDHybrid(n=10, stencil=33, epsilon=0.05, band=1.5),
        Newton(scheme=Hybrid(n=10, boundary="interpolate"), max_iter=9),
        Newton(scheme=Hybrid(n=10), start=FDMonotone(n=10, stencil=9)),
        Newton(
            scheme=Hybrid(n=10),
            start=PoissonIteration(scheme=Standard9(n=10), max_iter=1),
        ),
    ],
    ids=["preset", "composed", "with a start", "with a Poisson start"],
)
def test_hparams_rebuild_the_solver(solver):
    pytest.importorskip("jsonargparse")
    assert from_config(solver.hparams) == solver
