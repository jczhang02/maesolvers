"""The torch kernels of the neural losses (needs the neural extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
functional = pytest.importorskip("maesolvers.neural.functional")

from maesolvers.core import Box, Disk, Ellipse  # noqa: E402

pytestmark = pytest.mark.neural


def _points(m=20, d=2, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(0, 1, (m, d)), dtype=torch.float64)


def _exp_half_square(x):
    return torch.exp(0.5 * (x**2).sum(dim=-1))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_hessian_of_a_known_function(d):
    x = _points(d=d)
    u = _exp_half_square(x)
    expected = (torch.eye(d) + x[:, :, None] * x[:, None, :]) * u[:, None, None]
    hess = functional.hessian(_exp_half_square, x)
    assert hess.shape == (20, d, d) and hess.dtype == torch.float64
    torch.testing.assert_close(hess, expected, rtol=1e-12, atol=1e-12)


def test_hessian_of_an_affine_function_is_zero():
    x = _points()
    hess = functional.hessian(lambda p: p @ torch.tensor([1.0, 2.0], dtype=p.dtype), x)
    assert torch.equal(hess, torch.zeros(20, 2, 2, dtype=torch.float64))


def test_hessian_of_a_function_linear_in_one_coordinate():
    x = _points()
    hess = functional.hessian(lambda p: p[:, 0] ** 2 + 3.0 * p[:, 1], x)
    expected = torch.zeros(20, 2, 2, dtype=torch.float64)
    expected[:, 0, 0] = 2.0
    torch.testing.assert_close(hess, expected)


def test_hessian_keeps_the_graph_to_the_parameters():
    weight = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    x = _points()
    hess = functional.hessian(lambda p: weight * (p**2).sum(dim=-1), x)
    (grad,) = torch.autograd.grad(hess.sum(), weight)
    # The Hessian is 2 w I at each of the 20 points: d/dw of its sum is 2 * 2 * 20.
    assert grad.item() == pytest.approx(80.0)
    frozen = functional.hessian(_exp_half_square, x, create_graph=False)
    assert not frozen.requires_grad


def test_hessian_does_not_change_the_points():
    x = _points()
    before = x.clone()
    functional.hessian(_exp_half_square, x)
    assert torch.equal(x, before) and not x.requires_grad


def test_monge_ampere_residual_is_the_determinant_minus_f():
    hess = torch.tensor([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 0.0], [0.0, 1.0]]])
    residual = functional.monge_ampere_residual(hess, torch.tensor([5.0, 2.0]))
    torch.testing.assert_close(residual, torch.tensor([0.0, -1.0]))


def test_the_diagonal_penalty_is_the_mean_of_the_negative_parts_squared():
    hess = torch.tensor([[[-1.0, 5.0], [5.0, 2.0]], [[-2.0, 0.0], [0.0, -3.0]]])
    # (1 + 0) and (4 + 9), averaged over the two points.
    assert functional.diagonal_convexity_penalty(hess).item() == pytest.approx(7.0)
    convex = torch.eye(2).expand(4, 2, 2)
    assert functional.diagonal_convexity_penalty(convex).item() == 0.0


def test_the_misfit_is_a_mean_square_and_refuses_no_points():
    values = torch.tensor([1.0, 3.0])
    assert functional.mean_squared_misfit(values, torch.zeros(2)).item() == 5.0
    empty = torch.zeros(0)
    with pytest.raises(ValueError, match="empty tensors"):
        functional.mean_squared_misfit(empty, empty)


def test_gradient_of_a_known_function():
    x = _points(d=3)
    grad = functional.gradient(_exp_half_square, x)
    expected = x * _exp_half_square(x)[:, None]
    torch.testing.assert_close(grad, expected, rtol=1e-12, atol=1e-12)
    assert grad.requires_grad and not x.requires_grad


def test_gradient_keeps_the_graph_to_the_parameters_unless_asked_not_to():
    weight = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    x = _points()
    grad = functional.gradient(lambda p: weight * (p**2).sum(dim=-1), x)
    (dw,) = torch.autograd.grad(grad.sum(), weight)
    assert dw.item() == pytest.approx(2.0 * x.sum().item())
    frozen = functional.gradient(_exp_half_square, x, create_graph=False)
    assert not frozen.requires_grad


def test_gradient_of_a_network_that_ignores_the_points_is_zero():
    bias = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    x = _points()
    grad = functional.gradient(lambda p: bias.expand(len(p)), x)
    assert torch.equal(grad, torch.zeros_like(x))


TARGETS = [
    Box((-0.5, -0.5), (0.5, 0.5)),
    Disk((1.0, 2.0), 0.7),
    Ellipse((3.5, 0.0), (2.0, 0.5)),
]


@pytest.mark.parametrize("target", TARGETS, ids=repr)
def test_signed_distance_in_torch_is_the_numpy_one_and_trains(target):
    from maesolvers.core import closest_boundary_point, signed_distance

    rng = np.random.default_rng(5)
    mid = (np.asarray(target.bbox.lo) + np.asarray(target.bbox.hi)) / 2
    points = mid + rng.uniform(-3.0, 3.0, (200, 2))
    foot, _ = closest_boundary_point(target, points)
    points = np.concatenate([points, foot])  # on the boundary too
    _, normal = closest_boundary_point(target, points)
    y = torch.tensor(points, requires_grad=True)
    d = functional.signed_distance(target, y)
    assert d.dtype == torch.float64 and d.shape == (400,)
    np.testing.assert_array_equal(d.detach().numpy(), signed_distance(target, points))
    (grad,) = torch.autograd.grad(d.pow(2).sum() + d.sum(), y)
    # d(d^2 + d)/dy = (2 d + 1) nu: finite on the boundary, where it is nu.
    expected = (2 * d.detach().numpy() + 1)[:, None] * normal
    np.testing.assert_allclose(grad.numpy(), expected, atol=1e-12)


def test_signed_distance_in_torch_near_the_major_axis_of_an_ellipse():
    """Rounding noise off the axis of a symmetric problem leaves the loss finite."""
    from maesolvers.core import signed_distance

    target = Ellipse((3.5, 0.0), (2.0, 0.5))  # T12's target
    noise = np.array([1e-300, -1e-20, 1e-17, -1e-16, 1e-15])
    points = np.stack([np.full(5, 3.5 + 0.3), noise], axis=-1)
    y = torch.tensor(points, requires_grad=True)
    d = functional.signed_distance(target, y)
    np.testing.assert_array_equal(d.detach().numpy(), signed_distance(target, points))
    # On the axis, 0.3 from the centre, inside the evolute, the nearest point
    # (0.32, 0.5 sqrt(1 - 0.16^2)) leaves the axis: 0.16 = 2 * 0.3 / (2^2 - 0.5^2).
    on_axis = -np.sqrt((0.32 - 0.3) ** 2 + 0.25 * (1 - 0.16**2))
    np.testing.assert_allclose(d.detach().numpy(), on_axis, rtol=1e-14)
    (grad,) = torch.autograd.grad(d.pow(2).sum(), y)
    assert torch.isfinite(grad).all()
