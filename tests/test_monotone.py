"""The monotone wide-stencil scheme, its pure kernels and its preset."""

import dataclasses
import itertools

import numpy as np
import pytest
import scipy.sparse as sp

from maesolvers import (
    SOLVERS,
    T1,
    T2,
    T7,
    T8,
    T9,
    T12,
    T20,
    T22,
    Box,
    BoxGrid,
    Dirichlet,
    FDFiltered,
    FDHybrid,
    FDMonotone,
    Filtered,
    Hybrid,
    Monotone17,
    Newton,
    PoissonIteration,
    Problem,
    Standard9,
    UnsupportedProblemError,
    from_config,
)
from maesolvers.classical.monotone import (
    angular_resolution,
    directional_difference,
    frame_jacobian,
    frame_jacobian_coefficients,
    frame_values,
    monotone_jacobian,
    monotone_kappa,
    monotone_operator,
    stencil_dimension,
    stencil_frames,
)
from maesolvers.metrics import evaluate

HESSIAN = np.array([[2.0, 0.3], [0.3, 1.0]])
"""A Hessian whose eigenvectors are not grid directions, so no pair ties."""


def _quadratic(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.einsum("...i,ij,...j->...", x, HESSIAN, x) - x[..., 0]


def _constant(value):
    return lambda x: np.full(np.shape(x)[:-1], value)


def _grid(n, lo=0.0, side=1.0, dim=2):
    return BoxGrid(Box((lo,) * dim, (lo + side,) * dim), n)


def _interior_nodes(grid):
    return grid.nodes[grid.interior]


@pytest.mark.parametrize(
    ("points", "dtheta"),
    [(9, np.pi / 8), (17, np.arctan(0.5) / 2), (33, np.arctan(1 / 3) / 2)],
)
def test_angular_resolution_is_half_the_largest_gap_between_directions(points, dtheta):
    assert angular_resolution(points) == pytest.approx(dtheta, rel=1e-14)
    assert Monotone17(stencil=points).angular_resolution == angular_resolution(points)
    # Oberman 2008, eq. (3.3): the largest distance from an angle to the stencil.
    angles = np.linspace(0, np.pi, 100_001)
    directions = stencil_frames(points).reshape(-1, 2)
    theta = np.arctan2(directions[:, 1], directions[:, 0])
    gap = np.abs(np.angle(np.exp(2j * (angles[:, None] - theta)))) / 2
    assert np.max(np.min(gap, axis=1)) == pytest.approx(dtheta, abs=1e-4)


@pytest.mark.parametrize(("points", "pairs"), [(9, 2), (17, 4), (33, 8)])
def test_stencil_frames_are_orthogonal_coprime_grid_directions(points, pairs):
    found = stencil_frames(points)
    assert found.shape == (pairs, 2, 2)
    np.testing.assert_array_equal(found[0], [[1, 0], [0, 1]])
    assert np.all(np.einsum("pi,pi->p", found[:, 0], found[:, 1]) == 0)
    assert all(np.gcd(*direction) == 1 for direction in found.reshape(-1, 2))
    # every direction of the stencil once, up to sign: 1 + 2 * directions points
    directions = {
        tuple(d) if tuple(d) > (0, 0) else tuple(-d) for d in found.reshape(-1, 2)
    }
    assert 1 + 2 * len(directions) == points


@pytest.mark.parametrize("points", [8, 25, 27, 65])
def test_stencil_frames_rejects_other_sizes(points):
    with pytest.raises(ValueError, match=r"9, 17 or 33 points \(2D\) or 19 points"):
        stencil_frames(points)
    with pytest.raises(ValueError, match="stencil must be"):
        stencil_dimension(points)


@pytest.mark.parametrize(
    "direction", [tuple(d) for d in stencil_frames(33).reshape(-1, 2)]
)
def test_directional_difference_is_exact_on_quadratics_near_the_boundary_too(direction):
    """Unequal arms are exact on quadratics as well, so every row is ``nu^T H nu / |nu|^2``."""
    grid = _grid(7, lo=-0.5, side=1.4)
    operator, data, full = directional_difference(grid, direction, _quadratic)
    v = _quadratic(_interior_nodes(grid))
    nu = np.array(direction, dtype=float)
    np.testing.assert_allclose(
        operator @ v + data, nu @ HESSIAN @ nu / (nu @ nu), rtol=1e-11
    )
    assert full.all() == (max(map(abs, direction)) == 1)


def test_directional_difference_shortens_the_arm_as_oberman_2008():
    """Direction (2, 1) one node off the boundary: 4 / (15 h^2) (u+ - 3 u0 + 2 u-)."""
    n, h = 6, 1.0 / 6
    grid = _grid(n)
    operator, data, full = directional_difference(grid, (2, 1), _constant(0.0))
    row = 0 * (n - 1) + 2  # node (1, 3): the backward arm reaches x = 0 at t = 1/2
    weights = operator.getrow(row).toarray().ravel()
    centre, forward = row, (3 - 1) * (n - 1) + 4 - 1  # node (3, 4)
    assert weights[centre] == pytest.approx(-3 * 4 / (15 * h * h))
    assert weights[forward] == pytest.approx(4 / (15 * h * h))
    assert np.count_nonzero(weights) == 2 and not full[row]
    _, data, _ = directional_difference(grid, (2, 1), lambda p: p[..., 1])
    assert data[row] == pytest.approx(2 * 4 / (15 * h * h) * (3 - 0.5) * h)


def test_frame_values_are_the_papers_on_convex_pairs_and_negative_otherwise():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((3, 2, 50))
    kappa = rng.uniform(0.5, 2.0, 50)
    values = frame_values(d, kappa)
    convex = (d >= 0).all(axis=1)
    np.testing.assert_array_equal(values[convex], (d[:, 0] * d[:, 1])[convex])
    assert np.all(values[~convex] < 0)
    # monotone: raising any second difference never lowers the value
    for k in range(2):
        raised = d.copy()
        raised[:, k] += rng.uniform(0, 1, (3, 50))
        assert np.all(frame_values(raised, kappa) >= values)


def test_frame_jacobian_coefficients_are_the_derivatives_away_from_kinks():
    rng = np.random.default_rng(1)
    d = rng.uniform(-2.0, 2.0, (2, 2, 40))
    kappa = np.full(40, 1.5)
    coefficients = frame_jacobian_coefficients(d, kappa, floor=0.0)
    eps = 1e-7
    for k in range(2):
        step = np.zeros_like(d)
        step[:, k] = eps
        numeric = (frame_values(d + step, kappa) - frame_values(d - step, kappa)) / (
            2 * eps
        )
        smooth = (d[:, k] < 0) | (d[:, 1 - k] > 0)  # off the flat part of a^+ b^+
        np.testing.assert_allclose(
            coefficients[:, k][smooth], numeric[smooth], rtol=1e-6
        )
    floored = frame_jacobian_coefficients(np.zeros((1, 2, 1)), np.ones(1), floor=0.25)
    np.testing.assert_array_equal(floored, 0.25)


def test_the_monotone_operator_is_the_least_available_frame_and_its_jacobian():
    rng = np.random.default_rng(2)
    frames, m, k = 3, 20, 7
    operators = sp.csr_matrix(rng.standard_normal((2 * frames * m, k)))
    v = rng.standard_normal(k)
    d = (operators @ v).reshape(frames, 2, m)
    available = np.ones((frames, m), dtype=bool)
    available[2, :10] = False
    f = rng.uniform(0.0, 3.0, m)
    kappa = monotone_kappa(f)
    np.testing.assert_array_equal(
        kappa, np.sqrt(np.maximum(np.maximum(f, f.mean()), 1))
    )
    assert np.all(monotone_kappa(f, 3) >= 1.0)
    value, active = monotone_operator(d, available, kappa)
    expected = np.where(available, frame_values(d, kappa), np.inf)
    np.testing.assert_array_equal(value, expected.min(axis=0))
    assert np.all(active[:10] < 2)
    jacobian = monotone_jacobian(d, active, kappa, 0.0, operators)
    step, eps = rng.standard_normal(k), 1e-7
    numeric = (
        monotone_operator(
            (operators @ (v + eps * step)).reshape(d.shape), available, kappa
        )[0]
        - monotone_operator(
            (operators @ (v - eps * step)).reshape(d.shape), available, kappa
        )[0]
    ) / (2 * eps)
    np.testing.assert_allclose(jacobian @ step, numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_frame_jacobian_weights_each_directional_difference(dim):
    rng = np.random.default_rng(2)
    frames, m = 4, 6
    blocks = [sp.random(m, m, density=0.5, random_state=k) for k in range(frames * dim)]
    weights = rng.uniform(-1.0, 1.0, (frames, dim, m))
    jacobian = frame_jacobian(sp.csr_matrix(sp.vstack(blocks)), weights)
    v = rng.uniform(size=m)
    expected = sum(
        weights[p, k] * (blocks[p * dim + k] @ v)
        for p in range(frames)
        for k in range(dim)
    )
    assert jacobian.shape == (m, m)
    np.testing.assert_allclose(jacobian @ v, expected, rtol=1e-12)


QUADRATIC = Problem(
    "Q",
    Box((-0.5, -0.5), (0.9, 0.9)),
    _constant(float(np.linalg.det(HESSIAN))),
    Dirichlet(_quadratic),
    exact=_quadratic,
)


@pytest.mark.parametrize("boundary", ["narrow", "interpolate"])
def test_residual_of_a_quadratic_is_its_angular_error_only(boundary):
    """The minimum over grid pairs overestimates det H, and is exact for a diagonal H."""
    scheme = Monotone17(n=8, stencil=33, boundary=boundary)
    grid = scheme.discretize(QUADRATIC)
    residual = scheme.residual(grid, _quadratic(grid.nodes[grid.interior]))
    frames = stencil_frames(33).astype(float)
    curvature = np.einsum("pki,ij,pkj->pk", frames, HESSIAN, frames) / np.einsum(
        "pki,pki->pk", frames, frames
    )
    products = np.where(grid.available, curvature.prod(axis=1)[:, None], np.inf)
    expected = products.min(axis=0) - np.linalg.det(HESSIAN)
    assert np.all(expected > 0)
    np.testing.assert_allclose(residual, expected, rtol=1e-9)


def test_jacobian_matches_finite_differences_of_the_residual():
    scheme = Monotone17(n=7, stencil=17)
    grid = scheme.discretize(QUADRATIC)
    rng = np.random.default_rng(2)
    v = _quadratic(grid.nodes[grid.interior]) + 1e-4 * rng.standard_normal(36)
    jacobian = scheme.jacobian(grid, v)
    assert isinstance(jacobian, sp.csc_matrix)
    eps = 1e-7
    numeric = np.column_stack(
        [
            (scheme.residual(grid, v + eps * e) - scheme.residual(grid, v - eps * e))
            / (2 * eps)
            for e in np.eye(36)
        ]
    )
    np.testing.assert_allclose(jacobian.toarray(), numeric, rtol=1e-5, atol=1e-6)


def test_jacobian_is_an_m_matrix_on_a_non_convex_iterate():
    scheme = Monotone17(n=12, stencil=17)
    grid = scheme.discretize(T2)
    v = -scheme.solve_laplace(grid, np.ones(121))  # concave
    jacobian = scheme.jacobian(grid, v).toarray()
    off = jacobian - np.diag(np.diag(jacobian))
    assert np.all(np.diag(jacobian) < 0) and np.all(off >= 0)
    assert np.all(-np.diag(jacobian) >= off.sum(axis=1) * (1 - 1e-12))


def test_narrow_leaves_out_the_wide_pairs_next_to_the_boundary():
    grid = Monotone17(n=6, stencil=17).discretize(T1)
    available = grid.available.reshape(4, 5, 5)
    assert available[:2].all()  # the 9-point pairs everywhere
    assert available[2:, 1:-1, 1:-1].all() and not available[2:, 0].any()
    assert Monotone17(n=6, boundary="interpolate").discretize(T1).available.all()


def test_the_nine_point_stencil_never_leaves_the_box():
    narrow = FDMonotone(n=12, stencil=9).solve(T2)
    interpolate = FDMonotone(n=12, stencil=9, boundary="interpolate").solve(T2)
    np.testing.assert_array_equal(narrow.values, interpolate.values)


def test_solve_laplace_keeps_the_dirichlet_data():
    scheme = Monotone17(n=8)
    grid = scheme.discretize(QUADRATIC)
    v = scheme.solve_laplace(grid, np.full(49, np.trace(HESSIAN)))
    np.testing.assert_allclose(v, _quadratic(grid.nodes[grid.interior]), atol=1e-12)


def _ridge(x):
    return np.abs(np.asarray(x, dtype=float)[..., 0] - 0.5)


RIDGE = Problem(
    "ridge",
    Box((0.0, 0.0), (1.0, 1.0)),
    _constant(0.0),
    Dirichlet(_ridge),
    exact=_ridge,
)
"""``f = 0`` with convex, non-harmonic data: the root is the convex envelope."""


def _available_differences(scheme, grid, v):
    d = scheme.directional_differences(grid, v)
    return d[np.broadcast_to(grid.available[:, None], d.shape)]


@pytest.mark.parametrize("n", [5, 6, 12, 31])
@pytest.mark.parametrize("stencil", [9, 17, 33])
def test_newton_reaches_the_convex_envelope_where_f_vanishes(n, stencil):
    """The ridge lies between nodes for odd ``n`` and on them for even ``n``."""
    scheme = Monotone17(n=n, stencil=stencil)
    result = Newton(scheme=scheme, max_iter=200).solve(RIDGE)
    assert result.info["converged"]
    grid = scheme.discretize(RIDGE)
    assert _available_differences(scheme, grid, result.values[grid.interior]).min() > (
        -1e-8
    )
    assert evaluate(result)["max"] < 1e-9


def _newton_from(solver, problem, grid, v):
    """Run the solver's step hook from ``v`` on ``grid`` until it converges."""
    for _ in range(solver.max_iter):
        step = solver.step(problem, grid, v)
        assert step is not None, "the line search stalled"
        v, measure = step
        if solver.converged(problem, grid, measure):
            return v
    raise AssertionError("Newton did not converge")


@pytest.mark.parametrize(("problem", "atol"), [(T20, 1e-9), (RIDGE, 1e-7)])
def test_the_root_depends_neither_on_kappa_nor_on_the_start(problem, atol):
    """Ten times ``kappa``, and a non-convex start, reach the same grid function.

    On the ridge ``f = 0`` everywhere and the Jacobian is nearly singular in the
    flat parts, so the residual tolerance pins the values to about ``1e-8``.
    """
    scheme = Monotone17(n=16, stencil=17)
    solver = Newton(scheme=scheme, max_iter=200)
    grid = scheme.discretize(problem)
    reference = solver.solve(problem).values[grid.interior]
    nonconvex = -scheme.solve_laplace(grid, np.ones(grid.f.size))  # superharmonic
    assert _available_differences(scheme, grid, nonconvex).min() < -0.1
    for context in (grid, dataclasses.replace(grid, kappa=10.0 * grid.kappa)):
        for start in (solver.initial_guess(problem, context), nonconvex):
            v = _newton_from(solver, problem, context, start)
            np.testing.assert_allclose(v, reference, rtol=0, atol=atol)


def test_kappa_is_at_least_one_so_the_stopping_rule_sees_a_negative_difference():
    for problem in (RIDGE, T20, T2):
        grid = Monotone17(n=8).discretize(problem)
        assert grid.kappa.min() >= 1.0
    grid = Monotone17(n=8).discretize(T2)
    expected = np.sqrt(np.maximum(np.maximum(grid.f, grid.f.mean()), 1.0))
    np.testing.assert_allclose(grid.kappa, expected)


def test_a_bump_in_the_flat_disk_solves_the_bare_scheme_but_not_monotone17():
    """Where ``f = 0`` the bare ``min a^+ b^+`` cannot see a negative difference."""
    scheme = Monotone17(n=16, stencil=17)
    grid = scheme.discretize(T20)
    v = FDMonotone(n=16, stencil=17).solve(T20).values[grid.interior]
    centre = int(np.argmin(np.linalg.norm(grid.nodes[grid.interior] - 0.5, axis=1)))
    assert grid.f[centre] == 0.0
    v[centre] += 1e-3
    d = scheme.directional_differences(grid, v)
    assert np.all(d[:, :, centre] < 0)
    bare = np.where(
        grid.available, np.maximum(d[:, 0], 0) * np.maximum(d[:, 1], 0), np.inf
    ).min(axis=0)
    assert bare[centre] - grid.f[centre] == 0.0
    assert scheme.residual(grid, v)[centre] <= d[:, :, centre].max() < 0


def test_the_centred_hessian_is_indefinite_in_the_flat_disk_of_t20():
    """``node_hessian`` is not the scheme's: its ``u_xy`` mixes the diagonal pairs.

    Every stencil direction is convex at the root, yet the centred 9-point Hessian
    has a negative eigenvalue at nodes of the disk where ``f = 0``, so
    ``nonconvex_fraction`` is not zero (``benchmarks/monotone17.md``, Finding 6).
    """
    scheme = Monotone17(n=16, stencil=17)
    result = Newton(scheme=scheme, max_iter=200).solve(T20)
    grid = scheme.discretize(T20)
    assert _available_differences(scheme, grid, result.values[grid.interior]).min() > (
        -1e-8
    )
    assert evaluate(result)["nonconvex_fraction"] > 0.05
    eigmin = np.linalg.eigvalsh(result.node_hessian[grid.interior])[:, 0]
    in_disk = np.linalg.norm(grid.nodes[grid.interior] - 0.5, axis=1) <= 0.2
    assert np.all(eigmin[~in_disk] > -1e-8) and eigmin[in_disk].min() < -1e-3


def test_a_negative_or_infinite_f_is_an_error_naming_the_problem():
    for value in (-1.0, np.inf, np.nan):
        bad = Problem(
            "bad", Box((0.0, 0.0), (1.0, 1.0)), _constant(value), Dirichlet(_ridge)
        )
        with pytest.raises(ValueError, match=r"finite f >= 0.*bad"):
            FDMonotone(n=4).solve(bad)


def test_a_coarse_grid_takes_a_spline_of_degree_n():
    """A spline needs degree + 1 knots per axis; n = 2 gives three."""
    scheme = Monotone17(n=2, spline_degree=5)
    grid = scheme.discretize(QUADRATIC)
    solution = scheme.solution(grid, _quadratic(grid.nodes[grid.interior]))
    assert scheme.spline_degree == 5
    point = np.array([[0.3, 0.2]])
    assert solution.u(point) == pytest.approx(_quadratic(point), abs=1e-12)
    assert FDMonotone(n=2).solve(QUADRATIC).info["converged"]


def test_solution_carries_the_centred_hessian_and_the_residual():
    scheme = Monotone17(n=8, spline_degree=2)
    grid = scheme.discretize(QUADRATIC)
    solution = scheme.solution(grid, _quadratic(grid.nodes[grid.interior]))
    np.testing.assert_allclose(
        solution.node_hessian[grid.interior],
        np.broadcast_to(HESSIAN, (49, 2, 2)),
        atol=1e-10,
    )
    assert np.all(np.isnan(solution.node_hessian[~grid.interior]))
    assert solution.unknowns == 49 and solution.info["h"] == pytest.approx(
        (0.175, 0.175)
    )
    assert solution.info["residual_max"] > 0  # the angular error of the stencil
    point = np.array([[0.1, 0.2]])
    assert solution.u(point) == pytest.approx(_quadratic(point), abs=1e-12)


def test_supports_a_square_dirichlet_box_only():
    scheme = Monotone17(stencil=17)
    assert scheme.supports(T1) and scheme.supports(T7)
    assert not scheme.supports(T8) and not scheme.supports(T12)
    oblong = Problem(
        "R", Box((0.0, 0.0), (2.0, 1.0)), _constant(1.0), Dirichlet(_constant(0.0))
    )
    cube = Problem(
        "C", Box((0,) * 3, (1,) * 3), _constant(1.0), Dirichlet(_constant(0.0))
    )
    assert not scheme.supports(oblong) and not scheme.supports(cube)
    with pytest.raises(UnsupportedProblemError, match="equal sides"):
        scheme.discretize(oblong)
    with pytest.raises(UnsupportedProblemError, match=r"FDMonotone .* equal sides"):
        FDMonotone(n=4).solve(oblong)
    with pytest.raises(UnsupportedProblemError, match="FDMonotone does not support T8"):
        FDMonotone(n=4).solve(T8)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (dict(n=1), ValueError, "n must be at least 2"),
        (dict(n=8.0), TypeError, "n must be an integer"),
        (dict(stencil=25), ValueError, "stencil must be 9, 17 or 33 points"),
        (dict(stencil=27), ValueError, r"or 19 points \(3D\)"),
        (dict(stencil="17"), TypeError, "stencil must be an integer"),
        (
            dict(boundary="wide"),
            ValueError,
            "boundary must be 'narrow' or 'interpolate'",
        ),
        (dict(spline_degree=6), ValueError, "between 1 and 5, got 6"),
        (dict(jacobian_floor=0.0), ValueError, "jacobian_floor must be positive"),
    ],
)
def test_monotone17_validates_its_parameters(kwargs, error, message):
    with pytest.raises(error, match=message):
        Monotone17(**kwargs)
    with pytest.raises(error, match=message):
        FDMonotone(**kwargs)


def test_monotone17_is_a_frozen_keyword_only_part():
    assert Monotone17(n=np.int64(8), stencil=np.int64(9)) == Monotone17(n=8, stencil=9)
    assert type(Monotone17(stencil=np.int64(9)).stencil) is int
    with pytest.raises(TypeError):
        Monotone17(8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Monotone17().n = 4


def test_the_poisson_iteration_needs_what_the_scheme_lacks():
    with pytest.raises(TypeError, match="needs a scheme with second_derivatives"):
        PoissonIteration(scheme=Monotone17(n=8)).solve(T1)


@pytest.mark.parametrize("problem", [T1, T2])
def test_the_preset_is_its_composition(problem):
    composed = Newton(scheme=Monotone17(n=12, stencil=33), max_iter=200)
    ours = FDMonotone(n=12, stencil=33)
    assert ours.scheme == composed.scheme and isinstance(ours, Newton)
    a, b = ours.solve(problem), composed.solve(problem)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.info["history"] == b.info["history"]


def test_the_preset_has_flat_hparams_and_its_scheme_defaults():
    assert FDMonotone().hparams == {
        "class_path": "maesolvers.FDMonotone",
        "init_args": {
            "n": 64,
            "stencil": None,
            "boundary": "narrow",
            "spline_degree": 3,
            "max_iter": 200,
            "jacobian_floor": 1e-12,
            "tol": 1e-10,
        },
    }
    assert FDMonotone().scheme == Monotone17()
    assert SOLVERS["FDMonotone"] is FDMonotone
    assert (FDMonotone.domains, FDMonotone.boundaries, FDMonotone.dims) == (
        Monotone17.domains,
        Monotone17.boundaries,
        Monotone17.dims,
    )


def test_wider_stencils_are_more_accurate_on_the_smooth_example():
    errors = [
        evaluate(FDMonotone(n=24, stencil=s).solve(T1))["max"] for s in (9, 17, 33)
    ]
    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize(
    "solver",
    [
        FDMonotone(n=10, stencil=33, boundary="interpolate"),
        Newton(scheme=Monotone17(n=10, stencil=9), max_iter=7),
    ],
    ids=["preset", "composed"],
)
def test_hparams_rebuild_the_solver_through_jsonargparse(solver):
    jsonargparse = pytest.importorskip("jsonargparse")
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(type(solver), "solver")
    config = parser.parse_object({"solver": solver.hparams["init_args"]})
    assert parser.instantiate(config).solver == solver


def test_a_misspelt_boundary_is_a_config_error():
    pytest.importorskip("jsonargparse")
    with pytest.raises(ValueError, match="boundary"):
        from_config({"class_path": "FDMonotone", "init_args": {"boundary": "wide"}})


def test_a_stencil_size_outside_the_choices_is_a_config_error():
    pytest.importorskip("jsonargparse")
    with pytest.raises(ValueError, match="stencil"):
        from_config({"class_path": "FDMonotone", "init_args": {"stencil": 25}})
    assert from_config({"class_path": "FDMonotone", "init_args": {"stencil": 33}}) == (
        FDMonotone(stencil=33)
    )


def test_the_jacobian_floor_changes_the_steps_not_the_solution():
    papers = FDMonotone(n=24, jacobian_floor=1e-8).solve(T20)
    ours = FDMonotone(n=24).solve(T20)
    assert papers.info["converged"] and ours.info["converged"]
    np.testing.assert_allclose(papers.values, ours.values, atol=1e-9)


# The 19-point stencil in 3D (Froese-Oberman 2011, SIAM J. Numer. Anal., Section 7)

HESSIAN3 = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.5]])
"""A 3D Hessian whose eigenvectors are not grid directions."""


def _quadratic3(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.einsum("...i,ij,...j->...", x, HESSIAN3, x) + x[..., 2]


QUADRATIC3 = Problem(
    "Q3",
    Box((-0.5,) * 3, (0.9,) * 3),
    _constant(float(np.linalg.det(HESSIAN3))),
    Dirichlet(_quadratic3),
    exact=_quadratic3,
)


def test_stencil_dimension_is_2_for_the_plane_stencils_and_3_for_19_points():
    assert [stencil_dimension(p) for p in (9, 17, 33, 19)] == [2, 2, 2, 3]
    assert [Monotone17(stencil=p).dim for p in (9, 17, 33, 19)] == [2, 2, 2, 3]


def test_the_19_point_frames_are_every_orthogonal_triple_of_width_one():
    """Brute force over the 13 directions of the 27-point cube, up to sign."""
    frames = stencil_frames(19)
    assert frames.shape == (4, 3, 3)
    np.testing.assert_array_equal(frames[0], np.eye(3, dtype=int))
    cube = [
        v
        for v in itertools.product((-1, 0, 1), repeat=3)
        if any(v) and next(c for c in v if c) > 0
    ]
    triples = {
        frozenset(t)
        for t in itertools.combinations(cube, 3)
        if all(np.dot(a, b) == 0 for a, b in itertools.combinations(t, 2))
    }
    assert {frozenset(map(tuple, f)) for f in frames} == triples
    # the axes and the six face diagonals: 1 + 2 * 9 = 19 points
    directions = {tuple(d) for d in frames.reshape(-1, 3)}
    assert len(directions) == 9 and 1 + 2 * len(directions) == 19
    assert all(sum(map(abs, d)) <= 2 for d in directions)


def test_the_angular_resolution_of_19_points_is_the_body_diagonals_angle():
    dtheta = angular_resolution(19)
    assert dtheta == pytest.approx(np.arccos(np.sqrt(2 / 3)), rel=1e-12)
    assert Monotone17(stencil=19).angular_resolution == dtheta
    # no direction of the sphere is farther from the stencil
    rng = np.random.default_rng(3)
    lines = rng.standard_normal((20_000, 3))
    lines /= np.linalg.norm(lines, axis=1, keepdims=True)
    directions = stencil_frames(19).reshape(-1, 3).astype(float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    nearest = np.arccos(np.clip(np.abs(lines @ directions.T).max(axis=1), 0, 1))
    assert nearest.max() <= dtheta and nearest.max() > dtheta - 0.02


@pytest.mark.parametrize(
    "direction", [tuple(d) for d in stencil_frames(19)[:, :, :].reshape(-1, 3)]
)
def test_directional_difference_in_3d_is_exact_on_quadratics(direction):
    grid = _grid(5, lo=-0.5, side=1.4, dim=3)
    operator, data, full = directional_difference(grid, direction, _quadratic3)
    v = _quadratic3(_interior_nodes(grid))
    nu = np.array(direction, dtype=float)
    np.testing.assert_allclose(
        operator @ v + data, nu @ HESSIAN3 @ nu / (nu @ nu), rtol=1e-11
    )
    assert full.all()  # an arm of width 1 always ends at a node


def test_directional_difference_in_3d_shortens_an_arm_that_leaves_the_box():
    """Direction (2, 0, 1) from node (3, 1, 1) of 4 cells: the forward arm stops at x = 1."""
    grid = _grid(4, dim=3)
    _, _, full = directional_difference(grid, (2, 0, 1), _quadratic3)
    _, data, _ = directional_difference(grid, (2, 0, 1), _constant(1.0))
    row = (3 - 1) * 9 + (1 - 1) * 3 + (1 - 1)
    assert not full[row] and data[row] > 0


def test_frame_values_in_3d_are_the_product_on_convex_frames():
    rng = np.random.default_rng(4)
    d = rng.standard_normal((4, 3, 60))
    kappa = rng.uniform(1.0, 2.0, 60)
    values = frame_values(d, kappa)
    convex = (d >= 0).all(axis=1)
    np.testing.assert_allclose(values[convex], d.prod(axis=1)[convex], rtol=1e-15)
    assert np.all(values[~convex] < 0)
    for k in range(3):
        raised = d.copy()
        raised[:, k] += rng.uniform(0, 1, (4, 60))
        assert np.all(frame_values(raised, kappa) >= values)


def test_frame_jacobian_coefficients_in_3d_are_the_derivatives_away_from_kinks():
    rng = np.random.default_rng(5)
    d = rng.uniform(-2.0, 2.0, (4, 3, 40))
    kappa = np.full(40, 1.5)
    coefficients = frame_jacobian_coefficients(d, kappa, floor=0.0)
    eps = 1e-7
    for k in range(3):
        step = np.zeros_like(d)
        step[:, k] = eps
        numeric = (frame_values(d + step, kappa) - frame_values(d - step, kappa)) / (
            2 * eps
        )
        others = np.delete(d, k, axis=1)
        smooth = (d[:, k] < 0) | (others > 0).all(axis=1)
        np.testing.assert_allclose(
            coefficients[:, k][smooth], numeric[smooth], rtol=1e-6
        )
    floored = frame_jacobian_coefficients(np.zeros((1, 3, 1)), np.ones(1), floor=0.25)
    np.testing.assert_array_equal(floored, 0.25**2)


def test_residual_of_a_3d_quadratic_is_its_angular_error_only():
    """The minimum over the four triples overestimates det H (Hadamard)."""
    scheme = Monotone17(n=5, stencil=19)
    grid = scheme.discretize(QUADRATIC3)
    residual = scheme.residual(grid, _quadratic3(grid.nodes[grid.interior]))
    frames = stencil_frames(19).astype(float)
    curvature = np.einsum("pki,ij,pkj->pk", frames, HESSIAN3, frames) / np.einsum(
        "pki,pki->pk", frames, frames
    )
    expected = curvature.prod(axis=1).min() - np.linalg.det(HESSIAN3)
    assert expected > 0
    np.testing.assert_allclose(residual, expected, rtol=1e-9)
    assert grid.available.all()  # 19 points never leave the box


def test_the_3d_jacobian_matches_finite_differences_of_the_residual():
    scheme = Monotone17(n=4, stencil=19)
    grid = scheme.discretize(QUADRATIC3)
    rng = np.random.default_rng(6)
    v = _quadratic3(grid.nodes[grid.interior]) + 1e-4 * rng.standard_normal(27)
    jacobian = scheme.jacobian(grid, v)
    eps = 1e-7
    numeric = np.column_stack(
        [
            (scheme.residual(grid, v + eps * e) - scheme.residual(grid, v - eps * e))
            / (2 * eps)
            for e in np.eye(27)
        ]
    )
    np.testing.assert_allclose(jacobian.toarray(), numeric, rtol=1e-5, atol=1e-6)


def test_the_3d_jacobian_is_an_m_matrix_on_a_non_convex_iterate():
    scheme = Monotone17(n=6, stencil=19)
    grid = scheme.discretize(T22)
    v = -scheme.solve_laplace(grid, np.ones(125))  # concave
    jacobian = scheme.jacobian(grid, v).toarray()
    off = jacobian - np.diag(np.diag(jacobian))
    assert np.all(np.diag(jacobian) < 0) and np.all(off >= 0)
    assert np.all(-np.diag(jacobian) >= off.sum(axis=1) * (1 - 1e-12))


def test_kappa_in_3d_is_the_two_thirds_power():
    grid = Monotone17(n=6, stencil=19).discretize(T22)
    scale = np.maximum(np.maximum(grid.f, grid.f.mean()), 1.0)
    np.testing.assert_allclose(grid.kappa, scale ** (2 / 3), rtol=1e-15)
    assert grid.kappa.min() >= 1.0


def test_solve_laplace_in_3d_keeps_the_dirichlet_data():
    scheme = Monotone17(n=5, stencil=19)
    grid = scheme.discretize(QUADRATIC3)
    v = scheme.solve_laplace(grid, np.full(64, np.trace(HESSIAN3)))
    np.testing.assert_allclose(v, _quadratic3(grid.nodes[grid.interior]), atol=1e-12)


def test_a_3d_solution_carries_the_centred_hessian_and_the_spacing():
    scheme = Monotone17(n=5, stencil=19)
    grid = scheme.discretize(QUADRATIC3)
    solution = scheme.solution(grid, _quadratic3(grid.nodes[grid.interior]))
    np.testing.assert_allclose(
        solution.node_hessian[grid.interior],
        np.broadcast_to(HESSIAN3, (64, 3, 3)),
        atol=1e-10,
    )
    assert solution.unknowns == 64
    assert solution.info["h"] == pytest.approx((0.28, 0.28, 0.28))
    point = np.array([[0.1, 0.2, -0.3]])
    assert solution.u(point) == pytest.approx(_quadratic3(point), abs=1e-12)


def _ridge3(x):
    return np.abs(np.asarray(x, dtype=float)[..., 0] - 0.5)


RIDGE3 = Problem(
    "ridge3",
    Box((0.0,) * 3, (1.0,) * 3),
    _constant(0.0),
    Dirichlet(_ridge3),
    exact=_ridge3,
)


@pytest.mark.parametrize("n", [5, 6])
def test_newton_reaches_the_convex_envelope_in_3d_where_f_vanishes(n):
    """Every frame has a factor zero on the ridge: the floor squared carries Newton."""
    scheme = Monotone17(n=n, stencil=19)
    result = Newton(scheme=scheme, max_iter=200).solve(RIDGE3)
    assert result.info["converged"]
    assert evaluate(result)["max"] < 1e-9


@pytest.mark.parametrize("problem", [T9, T22])
def test_fdmonotone_with_19_points_solves_the_3d_catalogue_problems(problem):
    result = FDMonotone(n=8, stencil=19).solve(problem)
    composed = Newton(scheme=Monotone17(n=8, stencil=19), max_iter=200).solve(problem)
    np.testing.assert_array_equal(result.values, composed.values)
    assert result.info["converged"] and evaluate(result)["unknowns"] == 7**3
    assert result.nodes.shape == (9**3, 3)
    assert 1e-3 < evaluate(result)["max"] < 2e-2  # the angular error of 19 points


def test_the_stencil_decides_the_dimension_a_scheme_supports():
    assert Monotone17(stencil=19).supports(T9) and Monotone17(stencil=19).supports(T22)
    assert not Monotone17(stencil=19).supports(T1)
    assert not Monotone17(stencil=17).supports(T9)
    oblong = Problem(
        "R3", Box((0,) * 3, (1, 1, 2)), _constant(1.0), Dirichlet(_constant(0.0))
    )
    assert not Monotone17(stencil=19).supports(oblong)
    with pytest.raises(UnsupportedProblemError, match=r"19-point stencil .* 3D Box"):
        Monotone17(n=4, stencil=19).discretize(T1)
    with pytest.raises(UnsupportedProblemError, match="the 17-point stencil is 2D"):
        FDMonotone(n=4, stencil=17).solve(T9)
    with pytest.raises(UnsupportedProblemError, match="the 19-point stencil is 3D"):
        FDMonotone(n=4, stencil=19).solve(T1)
    with pytest.raises(UnsupportedProblemError, match=r"FDMonotone .* equal sides"):
        FDMonotone(n=4, stencil=19).solve(oblong)


def test_hybrid_and_filtered_refuse_the_3d_stencil():
    """Their standard scheme, Standard9, is 2D."""
    with pytest.raises(ValueError, match="Hybrid's stencil must be 9, 17 or 33"):
        Hybrid(stencil=19)
    with pytest.raises(ValueError, match="FDHybrid's stencil must be 9, 17 or 33"):
        FDHybrid(stencil=19)
    with pytest.raises(ValueError, match="FDFiltered's stencil must be 9, 17 or 33"):
        FDFiltered(stencil=19)
    with pytest.raises(ValueError, match="got Standard9 in 2D and Monotone17 in 3D"):
        Filtered(accurate=Standard9(n=8), monotone=Monotone17(n=8, stencil=19))
    assert Hybrid.dims == FDHybrid.dims == FDFiltered.dims == (2,)


def test_the_19_point_stencil_is_a_config_choice():
    pytest.importorskip("jsonargparse")
    assert from_config(
        {"class_path": "FDMonotone", "init_args": {"stencil": 19, "n": 8}}
    ) == FDMonotone(stencil=19, n=8)


def test_the_default_stencil_follows_the_dimension():
    """17 points in 2D, the 19-point scheme in 3D, resolved when the problem is known."""
    scheme = Monotone17(n=4)
    assert scheme.stencil is None and scheme.dim is None
    assert scheme.supports(T1) and scheme.supports(T9)
    assert scheme.resolve(2) == Monotone17(n=4, stencil=17)
    assert scheme.resolve(3) == Monotone17(n=4, stencil=19)
    assert Monotone17(n=4, stencil=33).resolve(3).stencil == 33
    with pytest.raises(ValueError, match="no default stencil in 4D"):
        scheme.resolve(4)
    with pytest.raises(ValueError, match="no angular resolution"):
        _ = scheme.angular_resolution
    for problem, points in ((T1, 17), (T9, 19)):
        grid = scheme.discretize(problem)
        assert grid.frames == len(stencil_frames(points))
        explicit = Monotone17(n=4, stencil=points)
        v = grid.f**0.5
        np.testing.assert_array_equal(
            scheme.residual(grid, v), explicit.residual(explicit.discretize(problem), v)
        )
    np.testing.assert_array_equal(
        FDMonotone(n=8).solve(T1).values, FDMonotone(n=8, stencil=17).solve(T1).values
    )
    np.testing.assert_array_equal(
        FDMonotone(n=5).solve(T9).values, FDMonotone(n=5, stencil=19).solve(T9).values
    )
    oblong = Problem(
        "R3", Box((0,) * 3, (1, 1, 2)), _constant(1.0), Dirichlet(_constant(0.0))
    )
    assert not scheme.supports(oblong) and not scheme.supports(T8)
    with pytest.raises(UnsupportedProblemError, match="a 2D or 3D Box"):
        scheme.discretize(oblong)
