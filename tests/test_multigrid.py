"""The multigrid iteration of Liu-Froese-Oberman-Xiao 2017, its transfers and its preset."""

import dataclasses

import numpy as np
import pytest

from maesolvers import (
    T1,
    T2,
    T8,
    T9,
    T22,
    T28,
    T46,
    Box,
    BoxGrid,
    Callback,
    FDMultigrid,
    Multigrid,
    Newton,
    Scheme,
    Standard9,
    StandardBox,
    UnsupportedProblemError,
)
from maesolvers.classical.multigrid import (
    cubic_resample,
    half_weighting,
    interpolation_matrix,
    resample,
    sweep_groups,
)
from maesolvers.metrics import evaluate

# Sweep orders


@pytest.mark.parametrize(("n", "d"), [(5, 2), (6, 2), (4, 3), (5, 3)])
@pytest.mark.parametrize("ordering", ["lexicographic", "red-black"])
def test_groups_cover_every_interior_node_once(n, d, ordering):
    groups = sweep_groups(n, d, ordering)
    nodes = np.sort(np.concatenate(groups))
    grid = BoxGrid(Box((0.0,) * d, (1.0,) * d), n)
    np.testing.assert_array_equal(nodes, np.flatnonzero(grid.interior))


def test_red_black_colours_by_the_parity_of_the_index_sum():
    red, black = sweep_groups(4, 2, "red-black")
    assert all(sum(np.unravel_index(k, (5, 5))) % 2 == 0 for k in red)
    assert all(sum(np.unravel_index(k, (5, 5))) % 2 == 1 for k in black)


@pytest.mark.parametrize("problem", [T1, T9])
def test_lexicographic_groups_reproduce_the_sequential_sweep_exactly(problem):
    scheme = StandardBox(n=5)
    grid = scheme.discretize(problem)
    rng = np.random.default_rng(0)
    start = problem.exact(grid.grid.nodes) + 0.05 * rng.normal(
        size=len(grid.grid.nodes)
    )
    start[~grid.grid.interior] = grid.boundary_values
    f = np.zeros(len(start))
    f[grid.grid.interior] = grid.f
    sequential = start.copy()
    for node in np.flatnonzero(grid.grid.interior):  # flat order, one by one
        sequential[node] = scheme.relax(grid, sequential, f[[node]], np.array([node]))[
            0
        ]
    grouped = start.copy()
    for nodes in sweep_groups(5, problem.dim, "lexicographic"):
        grouped[nodes] = scheme.relax(grid, grouped, f[nodes], nodes)
    np.testing.assert_array_equal(grouped, sequential)


def test_sweep_groups_checks_its_arguments():
    with pytest.raises(ValueError, match="ordering"):
        sweep_groups(4, 2, "diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n"):
        sweep_groups(1, 2, "red-black")


# Transfers


def test_interpolation_matrix_injects_onto_a_nested_coarse_grid():
    matrix = interpolation_matrix(8, 4).toarray()
    np.testing.assert_array_equal(matrix, np.eye(9)[::2])


def test_interpolation_matrix_takes_midpoints_from_a_nested_coarse_grid():
    matrix = interpolation_matrix(2, 4).toarray()
    np.testing.assert_array_equal(
        matrix, [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]]
    )


@pytest.mark.parametrize(("n_from", "n_to"), [(7, 3), (3, 7), (12, 5), (5, 12)])
def test_interpolation_matrix_is_exact_on_linear_functions(n_from, n_to):
    matrix = interpolation_matrix(n_from, n_to)
    x_from, x_to = np.linspace(0, 1, n_from + 1), np.linspace(0, 1, n_to + 1)
    np.testing.assert_allclose(
        matrix @ (2.0 * x_from - 1.0), 2.0 * x_to - 1.0, atol=1e-15
    )
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


@pytest.mark.parametrize("d", [2, 3])
def test_resample_is_exact_on_multilinear_functions(d):
    def bilinear(n):
        axes = np.meshgrid(*[np.linspace(0, 1, n + 1)] * d, indexing="ij")
        return 1.0 + np.prod(axes, axis=0) + axes[0]

    np.testing.assert_allclose(resample(bilinear(4), 8), bilinear(8), atol=1e-14)
    np.testing.assert_allclose(resample(bilinear(8), 4), bilinear(4), atol=1e-14)
    np.testing.assert_allclose(resample(bilinear(6), 9), bilinear(9), atol=1e-14)


@pytest.mark.parametrize(("d", "face"), [(2, 1 / 8), (3, 1 / 12)])
def test_half_weighting_is_the_papers_stencil(d, face):
    r = np.zeros((7,) * d)
    centre = (3,) * d
    r[centre] = 1.0
    out = half_weighting(r)
    assert out[centre] == 0.5
    for k in range(d):
        for sign in (1, -1):
            index = list(centre)
            index[k] += sign
            assert out[tuple(index)] == pytest.approx(face)
    assert out.sum() == pytest.approx(0.5 + 2 * d * face)


def test_half_weighting_is_zero_on_the_ring():
    out = half_weighting(np.pad(np.ones((5, 5, 5)), 1))
    inner = (slice(1, -1),) * 3
    ring = np.ones(out.shape, dtype=bool)
    ring[inner] = False
    assert np.all(out[ring] == 0.0)


@pytest.mark.parametrize("d", [2, 3])
def test_cubic_resample_is_exact_on_cubics(d):
    def cubic(n):
        axes = np.meshgrid(*[np.linspace(0, 1, n + 1)] * d, indexing="ij")
        return sum(x**3 - 2 * x**2 for x in axes) + np.prod(axes, axis=0)

    np.testing.assert_allclose(cubic_resample(cubic(4), 8), cubic(8), atol=1e-13)


def test_cubic_resample_from_two_cells_is_quadratic():
    x = np.linspace(0, 1, 3)
    out = cubic_resample(np.outer(x**2, np.ones(3)), 4)
    np.testing.assert_allclose(out, np.outer(np.linspace(0, 1, 5) ** 2, np.ones(5)))


# The iteration


@pytest.mark.parametrize("ordering", ["lexicographic", "red-black"])
@pytest.mark.parametrize(("problem", "bound"), [(T1, 3.5e-4), (T9, 4.5e-4)])
def test_one_full_multigrid_cycle_reaches_the_discretisation_error(
    problem, bound, ordering
):
    result = Multigrid(scheme=StandardBox(n=16), ordering=ordering).solve(problem)
    assert result.info["iterations"] == 1 and result.info["converged"]
    assert result.info["history"][0] <= 1e-6
    assert evaluate(result)["max"] < bound


def test_the_cycles_converge_to_the_root_newton_reaches_from_there():
    tight = Multigrid(scheme=StandardBox(n=8), tol=1e-13, max_iter=40).solve(T9)
    assert tight.info["converged"] and tight.info["residual_max"] < 1e-7
    newton = Newton(scheme=StandardBox(n=8), start=Multigrid(scheme=StandardBox(n=8)))
    polished = newton.solve(T9)
    np.testing.assert_allclose(polished.values, tight.values, atol=1e-10)


def test_each_v_cycle_reduces_the_residual_mesh_independently():
    rates = []
    for n in (8, 16, 32):
        history = Multigrid(scheme=StandardBox(n=n), tol=1e-30, max_iter=4).solve(T1)
        rates.append(history.info["history"][3] / history.info["history"][2])
    assert max(rates) < 0.2


def test_it_reaches_the_convex_root_where_newton_does_not():
    """T2: Newton on the standard scheme reaches a non-convex root
    (benchmarks/standard_fd.md, Finding 2); the Gauss-Seidel smoother does not."""
    newton = evaluate(Newton(scheme=StandardBox(n=32)).solve(T2))
    multigrid = evaluate(Multigrid(scheme=StandardBox(n=32)).solve(T2))
    assert multigrid["nonconvex_fraction"] == 0.0 < newton["nonconvex_fraction"]


@pytest.mark.parametrize("n", [2, 3, 12, 13])
def test_any_number_of_cells_converges(n):
    result = Multigrid(scheme=StandardBox(n=n), max_iter=20).solve(T1)
    assert result.info["converged"]
    assert evaluate(result)["max"] < 2e-2


def test_singular_problems_converge_in_3d():
    for problem in (T22, T28, T46):
        result = Multigrid(scheme=StandardBox(n=8), ordering="red-black").solve(problem)
        assert (
            result.info["converged"] and evaluate(result)["nonconvex_fraction"] == 0.0
        )


def test_callbacks_see_the_relative_residual_of_every_cycle():
    class Record(Callback):
        def __init__(self):
            self.steps = []

        def on_step_end(self, solver, step, metrics):
            self.steps.append((step, set(metrics)))

    record = Record()
    result = Multigrid(scheme=StandardBox(n=8), tol=1e-12).solve(T1, callbacks=[record])
    assert record.steps == [
        (k + 1, {"relative_residual"}) for k in range(len(record.steps))
    ]
    assert len(record.steps) == result.info["iterations"] > 1


def test_the_result_is_the_schemes_solution_on_the_finest_grid():
    result = Multigrid(scheme=StandardBox(n=8)).solve(T9)
    assert result.nodes.shape == (729, 3) and result.info["unknowns"] == 343
    assert result.info["interpolant"] == "spline(3)"


# Composition checks


def test_it_needs_a_scheme_with_relax():
    with pytest.raises(TypeError, match="relax"):
        Multigrid(scheme=Standard9(n=8)).solve(T1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Cells(Scheme):
    """A scheme whose grid size is not called n."""

    cells: int = 4

    def supports(self, problem):
        return True

    def discretize(self, problem):
        return None

    def rhs(self, grid):
        return np.zeros(0)

    def solution(self, grid, v):
        raise NotImplementedError


def test_it_needs_a_scheme_with_a_field_n():
    with pytest.raises(TypeError, match="field n"):
        Multigrid(scheme=_Cells())


def test_it_supports_boxes_with_dirichlet_data_only():
    solver = Multigrid(scheme=StandardBox(n=4))
    assert solver.supports(T1) and solver.supports(T22)
    assert not solver.supports(T8)
    with pytest.raises(UnsupportedProblemError):
        solver.solve(T8)


def test_parameters_are_checked():
    scheme = StandardBox(n=4)
    with pytest.raises(ValueError, match="ordering"):
        Multigrid(scheme=scheme, ordering="diagonal")  # type: ignore[arg-type]
    for name, value in (
        ("pre_sweeps", -1),
        ("post_sweeps", -1),
        ("coarsest_sweeps", 0),
    ):
        with pytest.raises(ValueError, match=name):
            Multigrid(scheme=scheme, **{name: value})
    with pytest.raises(TypeError, match="pre_sweeps"):
        Multigrid(scheme=scheme, pre_sweeps=1.5)


# The preset


def test_the_preset_is_multigrid_on_standard_box():
    preset = FDMultigrid(n=16, spline_degree=5, ordering="red-black", tol=1e-8)
    composed = Multigrid(
        scheme=StandardBox(n=16, spline_degree=5), ordering="red-black", tol=1e-8
    )
    for name in ("scheme", "ordering", "pre_sweeps", "post_sweeps", "max_iter", "tol"):
        assert getattr(preset, name) == getattr(composed, name)
    np.testing.assert_array_equal(preset.solve(T1).values, composed.solve(T1).values)


def test_the_preset_records_flat_parameters():
    assert FDMultigrid().hparams == {
        "class_path": "maesolvers.FDMultigrid",
        "init_args": {
            "max_iter": 50,
            "tol": 1e-6,
            "ordering": "lexicographic",
            "pre_sweeps": 2,
            "post_sweeps": 2,
            "coarsest_sweeps": 1,
            "n": 64,
            "spline_degree": 3,
        },
    }


def test_the_preset_declares_the_support_of_its_scheme():
    assert (FDMultigrid.domains, FDMultigrid.boundaries, FDMultigrid.dims) == (
        StandardBox.domains,
        StandardBox.boundaries,
        StandardBox.dims,
    )
    solver = FDMultigrid(n=4)
    assert solver.supports(T1) and solver.supports(T9) and not solver.supports(T8)
