"""The input convex network and its preset (needs the neural extra).

Convexity is a property of the construction, so the tests draw parameters
at random, negative roots included, and check it on the module; training
runs a handful of steps on a few points, as in test_neural.py.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from maesolvers import BFGS, ICNN, T1, T6, T9, DirichletICNN, PenaltyLoss  # noqa: E402
from maesolvers.metrics import evaluate  # noqa: E402
from maesolvers.neural import Batch, UniformSampler, _bfgs  # noqa: E402
from maesolvers.neural import functional as F  # noqa: E402
from maesolvers.neural._torch import building  # noqa: E402

pytestmark = pytest.mark.neural

SMALL = {"interior_points": 60, "boundary_points": 24, "accelerator": "cpu"}


def _module(dim=2, width=10, depth=5, seed=0):
    problem = {2: T1, 3: T9}[dim]
    points = UniformSampler(interior_points=4, boundary_points=0).sample(
        problem, np.random.default_rng(0)
    )
    with building(seed):
        return ICNN(width=width, depth=depth).build(problem, points)


def _points(n, dim, seed, scale=2.0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-scale, scale, (n, dim)), dtype=torch.float64)


def test_the_paper_network_has_563_parameters_all_trainable():
    module = _module()
    count = sum(p.numel() for p in module.parameters())
    assert count == 563  # Nystrom-Vestberg 2023, Example 4.2
    assert all(p.requires_grad for p in module.parameters())
    # In d dimensions: depth (d + 1) w input weights and biases,
    # (depth - 1) w^2 + w roots, and d + 1 for the output's input weights.
    module3 = _module(dim=3, width=4, depth=3)
    assert sum(p.numel() for p in module3.parameters()) == 3 * 4 * 4 + 2 * 16 + 4 + 4


def test_the_hidden_weights_are_non_negative_whatever_the_roots():
    module = _module(seed=1)
    with torch.no_grad():
        for root in module.roots:
            root.mul_(-3.0)
    assert any((root < 0).any() for root in module.roots)
    assert all((weight >= 0).all() for weight in module.weights())


@pytest.mark.parametrize(("dim", "depth", "seed"), [(2, 5, 0), (2, 1, 1), (3, 3, 2)])
def test_the_network_is_convex_by_construction(dim, depth, seed):
    module = _module(dim=dim, depth=depth, seed=seed)
    with torch.no_grad():  # far from the standard normal start
        for parameter in module.parameters():
            parameter.mul_(3.0).add_(torch.randn_like(parameter))
    x, y = _points(2000, dim, seed), _points(2000, dim, seed + 100)
    t = torch.rand(2000, dtype=torch.float64)[:, None]
    with torch.no_grad():
        chord = t[:, 0] * module(x) + (1 - t[:, 0]) * module(y)
        between = module(t * x + (1 - t) * y)
    scale = chord.abs().max()
    assert (between <= chord + 1e-12 * scale).all()
    hess = F.hessian(module, x, create_graph=False)
    eigmin = torch.linalg.eigvalsh(hess)[:, 0]
    assert (eigmin >= -1e-12 * hess.abs().amax()).all()


def test_negating_a_root_leaves_the_network_unchanged():
    module = _module(seed=3)
    x = _points(50, 2, 3)
    with torch.no_grad():
        before = module(x)
        module.roots[0].neg_()
        assert torch.equal(module(x), before)


def _batch(points, boundary, f, f_boundary, g):
    return Batch(interior=points, f=f, boundary=boundary, g=g, f_boundary=f_boundary)


def test_the_equation_term_can_sum_over_the_boundary_points_too():
    interior = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float64)
    boundary = torch.tensor([[0.0, 0.5]], dtype=torch.float64)
    batch = _batch(
        interior,
        boundary,
        f=torch.tensor([1.0, 3.0], dtype=torch.float64),
        f_boundary=torch.tensor([2.0], dtype=torch.float64),
        g=torch.zeros(1, dtype=torch.float64),
    )

    def u(x):  # det D^2 u = 4 at every point, u = 1.25 at the boundary point
        return x[:, 0] ** 2 + x[:, 1] ** 2 + 1.0

    inside = PenaltyLoss(convexity_weight=0.0, boundary_weight=100.0)(u, batch)
    both = PenaltyLoss(
        convexity_weight=0.0, boundary_weight=100.0, equation_at_boundary=True
    )(u, batch)
    assert inside["equation"].item() == pytest.approx(3.0**2 + 1.0**2)
    assert both["equation"].item() == pytest.approx(3.0**2 + 1.0**2 + 2.0**2)
    assert both["boundary"].item() == pytest.approx(1.25**2)
    assert both["loss"].item() == pytest.approx(14.0 + 100.0 * 1.25**2)
    assert both["convexity"].item() == 0.0


def test_the_convexity_term_stays_on_the_interior_points():
    interior = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    boundary = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    batch = _batch(interior, boundary, torch.ones(1), torch.ones(1), torch.zeros(1))

    def u(x):  # u_xx = 6 x: negative only at x < 0, here at no point
        return x[:, 0] ** 3

    loss = PenaltyLoss(equation_at_boundary=True)
    assert loss(u, batch)["convexity"].item() == 0.0
    flipped = _batch(boundary, interior, torch.ones(1), torch.ones(1), torch.zeros(1))
    assert loss(lambda x: -u(x), flipped)["convexity"].item() == pytest.approx(144.0)


def test_the_equation_at_the_boundary_needs_boundary_points():
    interior = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    empty = torch.zeros((0, 2), dtype=torch.float64)
    batch = _batch(interior, empty, torch.ones(1), torch.zeros(0), torch.zeros(0))
    with pytest.raises(ValueError, match="equation_at_boundary but the sampler"):
        PenaltyLoss(equation_at_boundary=True)(lambda x: x.sum(dim=-1), batch)


@pytest.fixture(scope="module")
def t6_result():
    return DirichletICNN(max_steps=5, **SMALL).solve(T6)


def test_the_preset_trains_a_convex_network(t6_result):
    info = t6_result.info
    assert info["unknowns"] == 563 and info["iterations"] == 5
    assert set(info["loss_terms"]) == {"loss", "equation", "convexity", "boundary"}
    assert info["loss_terms"]["convexity"] == 0.0
    history = info["history"]
    assert history[-1] < history[0]
    assert "network" not in info  # nothing is fitted before training
    metrics = evaluate(t6_result)
    assert metrics["nonconvex_fraction"] == 0.0 and metrics["eigmin"] > 0


def test_the_preset_trains_in_3d_with_a_convex_network():
    result = DirichletICNN(max_steps=3, **SMALL).solve(T9)
    info = result.info
    assert info["iterations"] == 3 and np.isfinite(info["loss_terms"]["loss"])
    history = info["history"]
    assert all(np.isfinite(history)) and history[-1] < history[0]
    assert evaluate(result)["nonconvex_fraction"] == 0.0


def test_a_batch_moves_its_tensors_and_keeps_g_none():
    batch = _batch(
        torch.zeros((2, 2)), torch.zeros((1, 2)), torch.ones(2), torch.ones(1), None
    )
    moved = batch.to("meta")
    assert moved.g is None
    assert all(
        getattr(moved, name).device.type == "meta"
        for name in ("interior", "f", "boundary", "f_boundary")
    )
    assert batch.interior.device.type == "cpu"  # the batch itself is unchanged


def test_the_solver_passes_f_at_the_boundary_points():
    solver = DirichletICNN(**SMALL)
    context = solver.setup(T6)
    boundary = context.collocation.boundary
    np.testing.assert_array_equal(context.f_boundary, T6.f(boundary))
    batch = solver._batch(context)
    assert batch.f_boundary.shape == (24,) and batch.f_boundary.dtype == torch.float64


def _rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def test_bfgs_minimises_the_rosenbrock_function():
    x = torch.nn.Parameter(torch.tensor([-1.2, 1.0], dtype=torch.float64))
    optimizer = BFGS().build([x])

    def closure():
        optimizer.zero_grad()
        loss = _rosenbrock(x)
        loss.backward()
        return loss

    for _ in range(100):
        optimizer.step(closure)
    torch.testing.assert_close(x.detach(), torch.ones(2, dtype=torch.float64))


def test_bfgs_takes_the_iterates_of_scipys_bfgs():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    start = np.array([-1.2, 1.0])
    steps = 25
    reference = scipy_optimize.minimize(
        lambda v: float(_rosenbrock(torch.as_tensor(v))),
        start,
        jac=lambda v: scipy_optimize.rosen_der(v),
        method="BFGS",
        options={"maxiter": steps, "gtol": 0.0},
    )
    x = torch.nn.Parameter(torch.tensor(start, dtype=torch.float64))
    optimizer = BFGS().build([x])

    def closure():
        optimizer.zero_grad()
        loss = _rosenbrock(x)
        loss.backward()
        return loss

    for _ in range(steps):
        optimizer.step(closure)
    assert reference.nit == steps
    np.testing.assert_allclose(x.detach().numpy(), reference.x, rtol=1e-9)


def test_bfgs_keeps_the_secant_condition_and_a_symmetric_approximation():
    x = torch.nn.Parameter(torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64))
    a = torch.tensor([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    a = a.to(torch.float64)
    optimizer = BFGS().build([x])

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * x @ a @ x + torch.sin(x).sum()
        loss.backward()
        return loss

    for _ in range(2):
        optimizer.step(closure)
    state = optimizer.state[x]
    previous_x, previous_g = state["x"].copy(), state["g"].copy()
    optimizer.step(closure)  # updates H with the pair of the last step
    h = state["h"]
    s, y = state["x"] - previous_x, state["g"] - previous_g
    np.testing.assert_allclose(h @ y, s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(h, h.T, rtol=0, atol=1e-12)
    assert (np.linalg.eigvalsh(h) > 0).all()


def test_bfgs_does_nothing_at_a_stationary_point():
    x = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = BFGS().build([x])

    def closure():
        optimizer.zero_grad()
        loss = (x**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    assert torch.equal(x.detach(), torch.zeros(2, dtype=torch.float64))


def _failing_search(f, fprime, x, d, *args, **kwargs):
    """A line search that moves the parameters to a trial point, then fails."""
    f(x + 0.5 * d)
    raise _bfgs._LineSearchError


def test_bfgs_restores_the_start_after_a_failed_line_search(monkeypatch):
    monkeypatch.setattr(_bfgs, "_line_search_wolfe12", _failing_search)
    start = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    x = torch.nn.Parameter(start.clone())
    optimizer = BFGS().build([x])

    def closure():
        optimizer.zero_grad()
        loss = _rosenbrock(x)
        loss.backward()
        return loss

    optimizer.step(closure)
    assert torch.equal(x.detach(), start)


def test_bfgs_leaves_the_parameters_when_scipys_line_searches_fail():
    # The closure's gradient is the negative of the loss's: every direction
    # it calls a descent direction goes uphill, so both of SciPy's searches
    # fail, after evaluating trial points.
    start = torch.tensor([0.3, -0.5], dtype=torch.float64)
    x = torch.nn.Parameter(start.clone())
    optimizer = BFGS().build([x])
    trials = []

    def closure():
        optimizer.zero_grad()
        value = (x**2).sum()
        loss = 2 * value.detach() - value  # the value of x^2, the gradient of -x^2
        loss.backward()
        trials.append(x.detach().clone())
        return loss

    optimizer.step(closure)
    assert any(not torch.equal(trial, start) for trial in trials)
    assert torch.equal(x.detach(), start)


def test_a_failed_line_search_stops_the_training_as_stalled(monkeypatch):
    monkeypatch.setattr(_bfgs, "_line_search_wolfe12", _failing_search)
    result = DirichletICNN(max_steps=5, **SMALL).solve(T6)
    assert result.info["stop"] == "stalled"
    assert result.info["iterations"] == 0 and len(result.info["history"]) == 0
