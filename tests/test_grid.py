"""The n-dimensional box grid, its operators and its interpolant (issue #25)."""

import itertools

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from maesolvers import T9, T10, Box, BoxGrid, Result
from maesolvers.classical.grid import (
    _nd_spline,
    centred_hessian,
    reach,
    second_differences,
    stencil,
    tensor_spline,
)
from maesolvers.classical.monotone import directional_difference
from maesolvers.metrics import evaluate

BOXES = {
    1: Box((-1.0,), (2.0,)),
    2: Box((-1.0, 0.0), (2.0, 0.5)),
    3: Box((0.0, -0.5, 1.0), (1.0, 0.5, 3.0)),
    4: Box((0.0, 0.0, -1.0, 2.0), (1.0, 2.0, 0.0, 2.5)),
}


def _quadratic(d):
    """A quadratic with a full, non-diagonal Hessian ``A`` in ``d`` dimensions."""
    a = np.arange(1.0, d * d + 1).reshape(d, d)
    a = a + a.T + d * d * np.eye(d)
    b = np.linspace(-1.0, 1.0, d)

    def u(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * np.einsum("...i,ij,...j->...", x, a, x) + x @ b

    return u, a


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_nodes_are_in_c_order_over_the_box(d):
    grid = BoxGrid(BOXES[d], 4)
    assert grid.dim == d and grid.shape == (5,) * d
    assert grid.nodes.shape == (5**d, d)
    for k, x in enumerate(grid.axes):
        np.testing.assert_array_equal(x, np.linspace(BOXES[d].lo[k], BOXES[d].hi[k], 5))
        assert grid.h[k] == x[1] - x[0]
    for flat, index in enumerate(itertools.product(range(5), repeat=d)):
        expected = [grid.axes[k][i] for k, i in enumerate(index)]
        np.testing.assert_array_equal(grid.nodes[flat], expected)


def test_two_dimensional_nodes_keep_the_flat_index_of_01():
    grid = BoxGrid(BOXES[2], 6)
    i, j = 4, 2
    np.testing.assert_array_equal(
        grid.nodes[i * 7 + j], [grid.axes[0][i], grid.axes[1][j]]
    )


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_the_boundary_ring_is_on_the_faces_and_the_interior_off_them(d):
    box, n = BOXES[d], 5
    grid = BoxGrid(box, n)
    assert np.count_nonzero(grid.interior) == (n - 1) ** d
    assert np.count_nonzero(~grid.interior) == (n + 1) ** d - (n - 1) ** d
    assert np.all(box.boundary_distance(grid.nodes[~grid.interior]) == 0.0)
    assert np.all(box.interior(grid.nodes[grid.interior]))


@pytest.mark.parametrize("d", [2, 3, 4])
def test_interior_index_lists_the_unknowns_in_flat_order(d):
    grid = BoxGrid(BOXES[d], 4)
    index = grid.interior_index
    assert index.shape == (3**d, d)
    flat = index @ (5 ** np.arange(d - 1, -1, -1))
    np.testing.assert_array_equal(flat, np.flatnonzero(grid.interior))


@pytest.mark.parametrize("d", [2, 3, 4])
def test_offset_steps_to_the_neighbour_along_a_direction(d):
    grid = BoxGrid(BOXES[d], 6)
    rng = np.random.default_rng(d)
    for _ in range(10):
        direction = rng.integers(-1, 2, size=d)
        start = np.flatnonzero(grid.interior)
        end = start + grid.offset(direction)
        steps = (grid.nodes[end] - grid.nodes[start]) / np.array(grid.h)
        np.testing.assert_allclose(steps, np.broadcast_to(direction, steps.shape))


@pytest.mark.parametrize("d", [2, 3, 4])
def test_even_grids_nest_in_the_node_layout(d):
    fine, coarse = BoxGrid(BOXES[d], 8), BoxGrid(BOXES[d], 4)
    every_other = (slice(None, None, 2),) * d
    nested = fine.nodes.reshape(*fine.shape, d)[every_other].reshape(-1, d)
    np.testing.assert_allclose(nested, coarse.nodes, rtol=0.0, atol=1e-15)


def test_reach_is_one_inside_and_the_fraction_to_the_face_outside():
    grid = BoxGrid(BOXES[3], 4)
    t = grid.reach((2, 0, -1)).reshape(3, 3, 3)  # interior indices 1..3 per axis
    # From index i along x the step of 2 stays in [0, 4] when i <= 2, and the
    # step of -1 along z always does.
    np.testing.assert_array_equal(t[:2], 1.0)
    np.testing.assert_array_equal(t[2], 0.5)
    t = grid.reach((3, 0, 3)).reshape(3, 3, 3)
    assert t[0, 0, 0] == 1.0 and t[0, 0, 2] == pytest.approx(1 / 3)
    assert t[2, 0, 0] == pytest.approx(1 / 3)


def test_reach_matches_the_full_arms_of_the_17_point_stencil():
    x = np.linspace(-0.5, 0.9, 9)
    grid = BoxGrid(Box((-0.5, -0.5), (0.9, 0.9)), 8)
    for direction in [(1, 0), (1, 1), (2, 1), (-1, 2), (3, -1)]:
        *_, full = directional_difference(x, x, direction, lambda p: p[..., 0])
        both = (grid.reach(direction) == 1.0) & (
            grid.reach(np.negative(direction)) == 1.0
        )
        np.testing.assert_array_equal(full, both)


@pytest.mark.parametrize("bad", [(0, 0, 0), (1, 0), (1.0, 0.0, 0.0)])
def test_reach_rejects_a_bad_direction(bad):
    with pytest.raises(ValueError, match="direction"):
        BoxGrid(BOXES[3], 4).reach(bad)
    with pytest.raises(ValueError, match="direction"):
        reach(BoxGrid(BOXES[3], 4).interior_index, 4, bad)


def test_reach_of_the_grid_is_the_kernel_at_its_interior_nodes():
    grid = BoxGrid(BOXES[2], 6)
    for direction in [(1, 0), (2, -1), (-3, 3)]:
        np.testing.assert_array_equal(
            grid.reach(direction), reach(grid.interior_index, 6, direction)
        )
    # On the ring the step can leave the box at once.
    assert reach(np.array([[6, 3], [0, 3]]), 6, (1, 0)).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="index must be an int array"):
        reach(np.array([[1.0, 2.0]]), 6, (1, 0))


def test_grid_validates_its_fields_and_keeps_its_arrays_read_only():
    with pytest.raises(TypeError, match="Box"):
        BoxGrid((0.0, 1.0), 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be at least 2"):
        BoxGrid(BOXES[2], 1)
    with pytest.raises(TypeError, match="n must be an integer"):
        BoxGrid(BOXES[2], 4.0)  # type: ignore[arg-type]
    grid = BoxGrid(BOXES[3], np.int64(3))
    assert type(grid.n) is int
    for array in (grid.nodes, grid.interior, grid.interior_index, *grid.axes):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


@pytest.mark.parametrize(("d", "points"), [(1, 3), (2, 9), (3, 19), (4, 33)])
def test_second_differences_are_exact_on_quadratics(d, points):
    grid = BoxGrid(BOXES[d], 5)
    u, a = _quadratic(d)
    operators = second_differences(grid.n, grid.h)
    assert sorted(operators) == [(k, m) for k in range(d) for m in range(k, d)]
    values = u(grid.nodes)
    for (k, m), operator in operators.items():
        assert operator.shape == (4**d, 6**d)
        np.testing.assert_allclose(operator @ values, a[k, m], rtol=1e-9)
    union = sum(abs(op) for op in operators.values())
    assert union.getrow(0).nnz == points  # the 9-, 19- and 33-point stencils


def test_stencil_rejects_wide_offsets_and_mismatched_weights():
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        stencil(4, np.array([[2, 0]]), [1.0])
    with pytest.raises(ValueError, match="offsets of shape"):
        stencil(4, np.array([[1, 0], [0, 1]]), [1.0])
    with pytest.raises(ValueError, match="offsets of shape"):
        stencil(4, np.zeros((1, 0), dtype=int), [1.0])
    # A fractional offset would land on a wrong column; an integral float is
    # refused too, as BoxGrid.offset refuses it.
    for offsets in ([[0.5, 0.0]], [[1.0, 0.0]]):
        with pytest.raises(ValueError, match="integers"):
            stencil(4, np.array(offsets), [1.0])


@pytest.mark.parametrize(
    ("h", "match"),
    [
        ((), "at least one axis"),
        ((0.1, 0.0), r"h\[1\] must be positive"),
        ((np.inf,), r"h\[0\] must be positive"),
        ((np.nan, 0.1), r"h\[0\]"),
    ],
)
def test_second_differences_check_the_spacing(h, match):
    with pytest.raises(ValueError, match=match):
        second_differences(4, h)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_centred_hessian_is_exact_on_quadratics_and_nan_on_the_ring(d):
    grid = BoxGrid(BOXES[d], 4)
    u, a = _quadratic(d)
    hess = centred_hessian(grid, u(grid.nodes))
    assert hess.shape == (5**d, d, d)
    np.testing.assert_allclose(
        hess[grid.interior], np.broadcast_to(a, (3**d, d, d)), rtol=1e-9
    )
    assert np.all(np.isnan(hess[~grid.interior]))
    with pytest.raises(ValueError, match="shape"):
        centred_hessian(grid, u(grid.nodes)[:-1])


@pytest.mark.parametrize("d", [1, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_tensor_spline_interpolates_and_reproduces_its_polynomials(d, degree):
    grid = BoxGrid(BOXES[d], 6)
    coefficients = np.linspace(0.5, 1.5, d)

    def poly(x):  # degree `degree` along each axis
        return np.prod(1.0 + coefficients * x**degree, axis=-1)

    u = grid.interpolant(poly(grid.nodes), degree)
    np.testing.assert_allclose(u(grid.nodes), poly(grid.nodes), rtol=1e-12)
    rng = np.random.default_rng(degree)
    points = rng.uniform(BOXES[d].lo, BOXES[d].hi, size=(4, 5, d))
    assert u(points).shape == (4, 5)
    np.testing.assert_allclose(u(points), poly(points), rtol=1e-10)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_tensor_spline_in_3d_extends_the_2d_spline(degree):
    # Values constant along the third axis: the 3D spline (make_interp_spline
    # and NdBSpline) is the 2D one (FITPACK) in every plane, to rounding.
    x, y, z = np.linspace(0.0, 1.0, 9), np.linspace(-1.0, 2.0, 7), np.linspace(0, 1, 6)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    plane = np.exp(xx * yy) + np.sin(3.0 * xx)
    flat = RectBivariateSpline(x, y, plane, kx=degree, ky=degree)
    u3 = tensor_spline((x, y, z), np.repeat(plane[..., None], len(z), axis=-1), degree)
    points = np.random.default_rng(0).uniform([0, -1, 0], [1, 2, 1], size=(500, 3))
    np.testing.assert_allclose(
        u3(points), flat(points[:, 0], points[:, 1], grid=False), rtol=0.0, atol=1e-13
    )
    u2 = tensor_spline((x, y), plane.ravel(), degree)
    np.testing.assert_array_equal(
        u2(points[:, :2]), flat(points[:, 0], points[:, 1], grid=False)
    )


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_both_spline_paths_agree_on_2d_axes(degree):
    # tensor_spline keeps FITPACK in 2D; the NdBSpline path of every other
    # dimension gives the same spline on the box, to about 1e-15 relative.
    x, y = np.linspace(0.0, 1.0, 9), np.linspace(-1.0, 2.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.exp(xx * yy) + np.sin(3.0 * xx)
    flat = RectBivariateSpline(x, y, values, kx=degree, ky=degree)
    points = np.random.default_rng(1).uniform([0, -1], [1, 2], size=(500, 2))
    np.testing.assert_allclose(
        _nd_spline((x, y), values, degree)(points),
        flat(points[:, 0], points[:, 1], grid=False),
        rtol=0.0,
        atol=1e-14 * np.max(np.abs(values)),
    )


def _cubic(p):
    """``sum_k p_k^3``, which a cubic spline reproduces on the box."""
    return np.sum(p**3, axis=-1)


def test_off_the_box_fitpack_clamps_and_ndbspline_extends():
    # The documented difference of the two paths: the spline is defined on
    # the box only.
    x = np.linspace(0.0, 1.0, 9)
    grid2 = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1)
    grid3 = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)
    u2 = tensor_spline((x, x), _cubic(grid2), 3)
    u3 = tensor_spline((x, x, x), _cubic(grid3), 3)
    assert u2(np.array([1.5, 0.5])) == pytest.approx(_cubic(np.array([1.0, 0.5])))
    assert u3(np.array([1.5, 0.5, 0.0])) == pytest.approx(
        _cubic(np.array([1.5, 0.5, 0.0]))
    )


def test_tensor_spline_checks_degree_and_points():
    with pytest.raises(ValueError, match="at least one coordinate array"):
        tensor_spline((), np.zeros(1), 1)
    axes = (np.linspace(0, 1, 4),) * 3
    with pytest.raises(ValueError, match="degree"):
        tensor_spline(axes, np.zeros(64), 4)
    with pytest.raises(ValueError, match="degree"):
        tensor_spline(axes, np.zeros(64), 0)
    u = tensor_spline(axes, np.zeros(64), 3)
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        u(np.zeros((5, 2)))


def _exact_result(problem, n):
    """The result a grid method with exact node values would return."""
    grid = BoxGrid(problem.domain, n)
    values = problem.exact(grid.nodes)
    return Result(
        problem,
        grid.interpolant(values, 3),
        nodes=grid.nodes,
        values=values,
        node_hessian=centred_hessian(grid, values),
        interior=grid.interior,
    )


@pytest.mark.parametrize(("problem", "sizes"), [(T9, (8, 16)), (T10, (6, 12))])
def test_exact_node_values_of_t9_and_t10_give_zero_error(problem, sizes):
    errors = []
    for n in sizes:
        m = evaluate(_exact_result(problem, n))
        assert m["max"] == 0.0 and m["rms"] == 0.0
        assert m["nonconvex_fraction"] == 0.0
        # exp(|x|^2/2) has Hessian (I + x x^T) u, smallest eigenvalue u(x) >= 1.
        assert m["eigmin"] == pytest.approx(m["eigmin_exact"], rel=0.05)
        errors.append(m["max_grid"])
    assert 0.0 < errors[1] < errors[0] / 10  # the cubic spline converges fast
