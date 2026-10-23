"""The standard scheme on a 2D or 3D box and its kernels."""

import itertools

import numpy as np
import pytest

from maesolvers import (
    T1,
    T8,
    T9,
    T10,
    T22,
    Box,
    BoxGrid,
    Dirichlet,
    Newton,
    PoissonIteration,
    Problem,
    Standard9,
    StandardBox,
    UnsupportedProblemError,
)
from maesolvers.classical.grid import (
    centred_differences,
    centred_hessian,
    cofactors,
    determinant,
)
from maesolvers.classical.standard_box import smallest_root
from maesolvers.metrics import evaluate


def _entries(matrices):
    """The entries ``(k, l)``, ``k <= l``, of an array of symmetric matrices."""
    d = matrices.shape[-1]
    return {(k, m): matrices[:, k, m] for k in range(d) for m in range(k, d)}


def _spd(rng, count, d):
    """Random symmetric matrices with eigenvalues between 0.5 and 3, or any sign."""
    q, _ = np.linalg.qr(rng.normal(size=(count, d, d)))
    mu = rng.uniform(0.5, 3.0, size=(count, d))
    return np.einsum("nij,nj,nkj->nik", q, mu, q)


# Kernels


@pytest.mark.parametrize("d", [2, 3])
def test_centred_differences_are_the_centred_hessian(d):
    grid = BoxGrid(Box((0.0,) * d, (1.0, 2.0, 0.5)[:d]), 6)
    values = np.random.default_rng(0).normal(size=len(grid.nodes))
    ours = centred_differences(values.reshape(grid.shape), grid.h)
    reference = centred_hessian(grid, values)[grid.interior]
    for (k, m), entry in ours.items():
        np.testing.assert_allclose(entry.ravel(), reference[:, k, m], rtol=1e-12)


@pytest.mark.parametrize("d", [2, 3])
def test_determinant_and_cofactors_match_numpy(d):
    matrices = np.random.default_rng(1).normal(size=(20, d, d))
    matrices = matrices + matrices.transpose(0, 2, 1)
    a = _entries(matrices)
    np.testing.assert_allclose(determinant(a, d), np.linalg.det(matrices), rtol=1e-12)
    adjugate = np.linalg.det(matrices)[:, None, None] * np.linalg.inv(matrices)
    for (k, m), cof in cofactors(a, d).items():
        np.testing.assert_allclose(cof, adjugate[:, m, k], rtol=1e-10, atol=1e-12)


def test_kernels_refuse_other_dimensions():
    a = _entries(np.eye(4)[None])
    for kernel in (determinant, cofactors):
        with pytest.raises(ValueError, match="2 or 3"):
            kernel(a, 4)
    with pytest.raises(ValueError, match="2 or 3"):
        smallest_root(a, np.ones(1), 4)


@pytest.mark.parametrize("d", [2, 3])
def test_smallest_root_for_positive_c_leaves_a_positive_definite_matrix(d):
    rng = np.random.default_rng(2)
    matrices = _spd(rng, 200, d) + rng.normal(size=(200, 1, 1))
    a = _entries(matrices)
    c = rng.uniform(1e-6, 10.0, size=200)
    t = smallest_root(a, c, d)
    shifted = matrices - t[:, None, None] * np.eye(d)
    np.testing.assert_allclose(np.linalg.det(shifted), c, rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(shifted)[:, 0] > 0.0)


def test_smallest_root_in_3d_starts_from_a_guess_below_or_anywhere():
    rng = np.random.default_rng(3)
    matrices = _spd(rng, 50, 3)
    a = _entries(matrices)
    c = rng.uniform(0.1, 1.0, size=50)
    plain = smallest_root(a, c, 3)
    for start in (plain - 0.1, plain, plain + 5.0, np.full(50, -1e3)):
        np.testing.assert_allclose(
            smallest_root(a, c, 3, start=start), plain, rtol=1e-13
        )


def test_smallest_root_in_3d_resolves_a_nearly_triple_eigenvalue():
    """A smooth solution's local matrix is close to a multiple of I; c ~ h^6."""
    matrices = 2.0 * np.eye(3)[None] + 1e-9 * np.random.default_rng(4).normal(
        size=(10, 3, 3)
    )
    matrices = 0.5 * (matrices + matrices.transpose(0, 2, 1))
    c = np.full(10, 1e-13)
    t = smallest_root(_entries(matrices), c, 3)
    shifted = matrices - t[:, None, None] * np.eye(3)
    np.testing.assert_allclose(np.linalg.det(shifted), c, rtol=1e-8)


def test_smallest_root_for_nonpositive_c_is_the_smallest_real_root():
    rng = np.random.default_rng(5)
    matrices = rng.normal(size=(100, 3, 3))
    matrices = matrices + matrices.transpose(0, 2, 1)
    c = -rng.uniform(0.0, 5.0, size=100)
    t = smallest_root(_entries(matrices), c, 3)
    for matrix, value, root in zip(matrices, c, t, strict=True):
        poly = np.poly(matrix)  # det(t I - A) = -det(A - t I) in 3D
        roots = np.roots(poly + np.array([0.0, 0.0, 0.0, value]))
        real = roots[np.abs(roots.imag) < 1e-7].real
        assert root == pytest.approx(real.min(), abs=1e-6)


def test_smallest_root_in_2d_is_the_vertex_without_a_real_root():
    a = {
        (0, 0): np.array([2.0, 2.0]),
        (1, 1): np.array([4.0, 4.0]),
        (0, 1): np.zeros(2),
    }
    # (2 - t)(4 - t) = c has real roots for c >= -1, none below.
    t = smallest_root(a, np.array([-0.75, -3.0]), 2)
    np.testing.assert_allclose(t, [2.5, 3.0])


# The scheme


def test_on_a_2d_box_the_equations_are_standard9s():
    rng = np.random.default_rng(6)
    ours, theirs = StandardBox(n=7), Standard9(n=7)
    grid, reference = ours.discretize(T1), theirs.discretize(T1)
    v = T1.exact(grid.grid.nodes[grid.grid.interior]) + 1e-3 * rng.normal(size=36)
    np.testing.assert_allclose(
        ours.residual(grid, v), theirs.residual(reference, v), rtol=1e-12, atol=1e-12
    )
    difference = ours.jacobian(grid, v) - theirs.jacobian(reference, v)
    assert abs(difference).max() < 1e-9 * abs(theirs.jacobian(reference, v)).max()
    np.testing.assert_allclose(
        ours.solve_laplace(grid, grid.f), theirs.solve_laplace(reference, reference.f)
    )


@pytest.mark.parametrize("problem", [T1, T9])
def test_jacobian_is_the_derivative_of_the_residual(problem):
    scheme = StandardBox(n=4)
    grid = scheme.discretize(problem)
    v = problem.exact(grid.grid.nodes[grid.grid.interior])
    jacobian = scheme.jacobian(grid, v).toarray()
    step = 1e-6
    for j in range(len(v)):
        e = np.zeros_like(v)
        e[j] = step
        column = (scheme.residual(grid, v + e) - scheme.residual(grid, v - e)) / (
            2 * step
        )
        np.testing.assert_allclose(jacobian[:, j], column, rtol=1e-5, atol=1e-5)


def test_solve_laplace_is_exact_on_quadratics():
    quadratic = Problem(
        "q",
        Box((0.0,) * 3, (1.0,) * 3),
        lambda x: np.ones(np.shape(x)[:-1]),
        Dirichlet(lambda x: 0.5 * np.sum(np.asarray(x) ** 2, axis=-1)),
    )
    scheme = StandardBox(n=5)
    grid = scheme.discretize(quadratic)
    v = scheme.solve_laplace(grid, np.full(64, 3.0))
    np.testing.assert_allclose(
        v, 0.5 * np.sum(grid.grid.nodes[grid.grid.interior] ** 2, axis=-1)
    )


def test_matrices_are_built_only_when_asked_for():
    scheme = StandardBox(n=8)
    grid = scheme.discretize(T9)
    scheme.residual(grid, np.zeros(343))
    assert "operators" not in vars(grid) and "laplace" not in vars(grid)
    scheme.solve_laplace(grid, grid.f)
    assert "operators" in vars(grid) and "laplace" in vars(grid)


def test_the_exact_solution_has_a_residual_of_order_h_squared():
    for problem in (T1, T9):
        scheme = StandardBox(n=16)
        grid = scheme.discretize(problem)
        v = problem.exact(grid.grid.nodes[grid.grid.interior])
        residual = np.max(np.abs(scheme.residual(grid, v)))
        coarse = StandardBox(n=8)
        coarse_grid = coarse.discretize(problem)
        w = problem.exact(coarse_grid.grid.nodes[coarse_grid.grid.interior])
        ratio = np.max(np.abs(coarse.residual(coarse_grid, w))) / residual
        assert 3.5 < ratio < 4.5


@pytest.mark.parametrize("d", [2, 3])
def test_relax_solves_each_nodes_equation_with_a_convex_hessian(d):
    problem = T1 if d == 2 else T9
    scheme = StandardBox(n=6)
    grid = scheme.discretize(problem)
    box = grid.grid
    u = problem.exact(box.nodes) + 0.01 * np.random.default_rng(7).normal(
        size=len(box.nodes)
    )
    u[~box.interior] = grid.boundary_values
    node = int(np.flatnonzero(box.interior)[len(grid.f) // 2])
    b = np.array([2.5])
    u[node] = scheme.relax(grid, u, b, np.array([node]))[0]
    hess = centred_hessian(box, u)[node]
    assert np.linalg.det(hess) == pytest.approx(2.5, rel=1e-10)
    assert np.all(np.linalg.eigvalsh(hess) > 0.0)


def test_relax_updates_a_group_from_the_values_it_is_given():
    scheme = StandardBox(n=6)
    grid = scheme.discretize(T9)
    u = T9.exact(grid.grid.nodes)
    nodes = np.flatnonzero(grid.grid.interior)[:5]
    together = scheme.relax(grid, u, grid.f[:5], nodes)
    one_by_one = [scheme.relax(grid, u, grid.f[[i]], nodes[[i]])[0] for i in range(5)]
    np.testing.assert_array_equal(together, one_by_one)


def test_solution_fields_and_info():
    scheme = StandardBox(n=8, spline_degree=5)
    grid = scheme.discretize(T22)
    v = T22.exact(grid.grid.nodes[grid.grid.interior])
    solution = scheme.solution(grid, v)
    assert solution.nodes.shape == (729, 3) and solution.node_hessian.shape == (
        729,
        3,
        3,
    )
    assert np.all(np.isnan(solution.node_hessian[~solution.interior]))
    assert solution.unknowns == 343
    assert solution.info["interpolant"] == "spline(5)"
    assert solution.info["h"] == (0.125,) * 3
    np.testing.assert_allclose(solution.u(grid.grid.nodes), solution.values, atol=1e-12)
    assert solution.info["residual_max"] == pytest.approx(
        np.max(np.abs(scheme.residual(grid, v)))
    )


def test_a_coarse_grid_takes_a_lower_spline_degree():
    grid = StandardBox(n=2).discretize(T9)
    assert grid.spline_degree == 2


def test_support_is_a_2d_or_3d_box_with_dirichlet_data():
    scheme = StandardBox()
    assert scheme.supports(T1) and scheme.supports(T9) and scheme.supports(T22)
    assert not scheme.supports(T8) and not scheme.supports(T10)
    with pytest.raises(
        UnsupportedProblemError, match="StandardBox does not support T10"
    ):
        scheme.discretize(T10)


def test_parameters_are_checked():
    with pytest.raises(ValueError, match="n"):
        StandardBox(n=1)
    with pytest.raises(TypeError, match="n"):
        StandardBox(n=2.5)
    with pytest.raises(ValueError, match="spline_degree"):
        StandardBox(spline_degree=6)


def test_newton_solves_it_and_poisson_refuses_it():
    result = Newton(scheme=StandardBox(n=8)).solve(T9)
    assert result.info["converged"]
    assert evaluate(result)["max"] < 2e-3
    with pytest.raises(TypeError, match="second_derivatives"):
        PoissonIteration(scheme=StandardBox(n=8)).solve(T1)


def test_unequal_spacing_is_the_scaled_equation():
    """On a box with unequal sides the root of relax still solves det H = b."""
    box = Box((0.0, 0.0, 0.0), (1.0, 2.0, 0.5))
    problem = Problem(
        "unequal",
        box,
        lambda x: np.ones(np.shape(x)[:-1]),
        Dirichlet(lambda x: 0.5 * np.sum(np.asarray(x) ** 2, axis=-1)),
    )
    scheme = StandardBox(n=4)
    grid = scheme.discretize(problem)
    u = 0.5 * np.sum(grid.grid.nodes**2, axis=-1)
    for node in itertools.islice(np.flatnonzero(grid.grid.interior), 3):
        assert scheme.relax(grid, u, np.ones(1), np.array([node]))[0] == pytest.approx(
            u[node], abs=1e-12
        )
