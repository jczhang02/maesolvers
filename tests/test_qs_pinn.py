"""QS-PINN: the QS form, its presets, and the ICNN plus a quadratic (needs the neural extra).

The structure of the QS form holds for any parameters (the note's
Proposition 3.3), so the tests draw them at random, far from the start, and
check it on the module; training runs a handful of steps on a few points, as
in test_icnn.py.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from maesolvers import (  # noqa: E402
    BFGS,
    ICNN,
    LBFGS,
    T1,
    T4,
    T8,
    T9,
    T12,
    T20,
    Adam,
    Box,
    Dirichlet,
    DirichletICNN,
    DirichletQSPINN,
    Disk,
    HausdorffLoss,
    NeuralSolver,
    PenaltyLoss,
    Problem,
    QuadraticForm,
    Switch,
    TransportICNN,
    TransportQSPINN,
    TransportSampler,
    UniformSampler,
    UnsupportedProblemError,
)
from maesolvers.metrics import evaluate  # noqa: E402
from maesolvers.neural import functional as F  # noqa: E402
from maesolvers.neural._torch import building  # noqa: E402

pytestmark = pytest.mark.neural

SMALL = {"interior_points": 40, "boundary_points": 20, "accelerator": "cpu"}


def _points(n, dim, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-scale, scale, (n, dim)), dtype=torch.float64)


def _module(network=None, problem=T8, seed=0):
    network = QuadraticForm() if network is None else network
    collocation = UniformSampler(interior_points=4, boundary_points=0).sample(
        problem, np.random.default_rng(0)
    )
    with building(seed):
        return network.build(problem, collocation)


def _scramble(module, seed=0, scale=3.0):
    """Move every parameter far from its start, the zero output layer included."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.mul_(scale).add_(noise.to(parameter.dtype))
    return module


# The kernels


def test_smooth_scaling_maps_into_the_open_frobenius_ball():
    z = torch.randn(500, 3, 3, dtype=torch.float64) * torch.logspace(-3, 3, 500)[
        :, None, None
    ].to(torch.float64)
    scaled = F.smooth_scaling(z, 0.7)
    norms = torch.linalg.matrix_norm(scaled)
    assert (norms < 0.7).all()
    expected = 0.7 * z / torch.sqrt(1 + torch.linalg.matrix_norm(z)[:, None, None] ** 2)
    torch.testing.assert_close(scaled, expected, rtol=1e-15, atol=0.0)
    small = torch.full((1, 2, 2), 1e-8, dtype=torch.float64)
    torch.testing.assert_close(F.smooth_scaling(small, 0.7), 0.7 * small)


def test_smooth_scaling_is_lipschitz_with_its_radius():
    a = torch.randn(200, 2, 2, dtype=torch.float64) * 3
    b = a + 1e-3 * torch.randn(200, 2, 2, dtype=torch.float64)
    change = torch.linalg.matrix_norm(
        F.smooth_scaling(a, 0.5) - F.smooth_scaling(b, 0.5)
    )
    assert (change <= 0.5 * torch.linalg.matrix_norm(a - b) * (1 + 1e-12)).all()


def test_the_quadratic_form_lies_above_its_tangent_plane_at_the_reference():
    x = _points(300, 2, 0, scale=2.0)
    matrix = torch.randn(300, 2, 2, dtype=torch.float64)
    reference = torch.tensor([0.3, -0.2], dtype=torch.float64)
    b = torch.tensor([1.5, -0.5], dtype=torch.float64)
    c = torch.tensor(0.25, dtype=torch.float64)
    u = F.quadratic_form(matrix, x, reference, b, c)
    v = x - reference
    expected = torch.einsum("mi,mji,mjk,mk->m", v, matrix, matrix, v) + x @ b + c
    torch.testing.assert_close(u, expected)
    assert (u >= x @ b + c - 1e-14).all()


def test_the_note_example_sqrt_1_plus_r2_has_an_exact_qs_form():
    # Example 4.5: sqrt(1 + |x|^2) = x^T m(|x|^2) x + 1, m(s) = 1 / (sqrt(1 + s) + 1),
    # so A = sqrt(m) I, b = 0, c = 1 (the note's P2, test-local: it has no T id).
    x = _points(400, 2, 1)
    s = x.pow(2).sum(dim=-1)
    m = 1 / (torch.sqrt(1 + s) + 1)
    matrix = torch.sqrt(m)[:, None, None] * torch.eye(2, dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64)
    u = F.quadratic_form(matrix, x, zero, zero, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(u, torch.sqrt(1 + s), rtol=1e-14, atol=1e-14)


# The network part


def test_the_form_starts_from_alpha_squared_times_the_distance_squared():
    module = _module()
    x = _points(50, 2, 2)
    with torch.no_grad():
        u = module(x)
    # T8: f = 1, so alpha = 1 / sqrt(2) and u = |x|^2 / 2, the note's P1 up to c.
    torch.testing.assert_close(u, 0.5 * x.pow(2).sum(dim=-1), rtol=1e-14, atol=1e-15)
    assert module.report == {
        "reference": [0.0, 0.0],
        "alpha_start": pytest.approx(2**-0.5, rel=1e-15),
        "alpha": pytest.approx(2**-0.5, rel=1e-15),
    }


def test_the_start_on_t12_maps_each_point_to_itself():
    module = _module(problem=T12)
    x = _points(50, 2, 3)
    torch.testing.assert_close(F.gradient(module, x, create_graph=False), x)


def test_alpha_starts_from_f_at_the_reference():
    module = _module(problem=T1)  # x0 = (1/2, 1/2), f(x0) = 3/2 e^(1/2)
    f0 = 1.5 * np.exp(0.5)
    assert module.report["alpha_start"] == pytest.approx(f0**0.25 / np.sqrt(2))
    x0 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    hess = F.hessian(module, x0, create_graph=False)
    assert float(torch.linalg.det(hess[0])) == pytest.approx(f0, rel=1e-12)


def test_the_paper_size_network_has_518_parameters_all_trainable():
    module = _module()
    assert sum(p.numel() for p in module.parameters()) == 518
    assert all(p.requires_grad for p in module.parameters())
    # d (w + 1) + (depth - 1)(w + 1) w + (w + 1) d^2 for the plain network,
    # then s (alpha = alpha_start exp(s)), b and c.
    module3 = _module(QuadraticForm(width=4, depth=2), problem=T9)
    assert sum(p.numel() for p in module3.parameters()) == 16 + 20 + 45 + 1 + 3 + 1


@pytest.mark.parametrize(("problem", "seed"), [(T8, 0), (T1, 1), (T9, 2)])
def test_the_structure_at_the_reference_holds_for_any_parameters(problem, seed):
    module = _scramble(_module(problem=problem), seed=seed)
    x0 = module.reference[None]
    with torch.no_grad():  # on the tangent plane c + b . x there
        plane = float(module.c + x0[0] @ module.b)
        assert float(module(x0)[0]) == pytest.approx(plane, rel=1e-14, abs=1e-14)
    torch.testing.assert_close(F.gradient(module, x0, create_graph=False)[0], module.b)
    a = module.matrix(x0)[0]
    hess = F.hessian(module, x0, create_graph=False)[0]
    torch.testing.assert_close(hess, 2 * a.T @ a, rtol=1e-12, atol=1e-12)
    assert float(torch.linalg.eigvalsh(hess)[0]) > 0
    x = _points(500, problem.dim, seed, scale=2.0)
    with torch.no_grad():
        assert (module(x) >= module.c + x @ module.b - 1e-12).all()


def test_the_matrix_is_invertible_with_singular_values_near_alpha():
    network = QuadraticForm(ratio=0.3)
    module = _scramble(_module(network), seed=4, scale=10.0)
    x = _points(1000, 2, 4, scale=3.0)
    with torch.no_grad():
        singular = torch.linalg.svdvals(module.matrix(x))
    alpha = float(module.alpha)
    assert alpha > 0
    assert (singular > alpha * 0.7).all() and (singular < alpha * 1.3).all()


@pytest.mark.parametrize("s", [-50.0, -3.0, 0.0, 4.0])
def test_alpha_is_positive_for_every_value_of_its_parameter(s):
    # u depends on alpha through alpha^2 only, so alpha = 0 (u = b . x + c, a
    # zero Hessian) would be a stationary point of every loss: the
    # parametrisation alpha = alpha_start exp(s) never reaches it.
    module = _scramble(_module(problem=T1), seed=5)
    with torch.no_grad():
        module.log_scale.fill_(s)
    alpha = float(module.alpha)
    assert alpha == pytest.approx(module.alpha_start * np.exp(s), rel=1e-14)
    assert alpha > 0
    assert module.report["alpha"] == alpha
    x0 = module.reference[None]
    a = module.matrix(x0)[0]
    assert float(torch.linalg.svdvals(a)[-1]) > 0.4 * alpha  # ratio 0.6


def test_the_report_keeps_alpha_current_when_a_wrapper_sets_it():
    module = _module()
    module.report = {**module.report, "identity_fit": {"steps": 0}, "alpha": -1.0}
    with torch.no_grad():
        module.log_scale.fill_(np.log(2.0))
    assert module.report == {
        "reference": [0.0, 0.0],
        "alpha_start": pytest.approx(2**-0.5, rel=1e-15),
        "identity_fit": {"steps": 0},
        "alpha": pytest.approx(2**0.5, rel=1e-15),
    }


def _f_at_the_centre(value):
    return Problem(
        "local",
        Box((0.0, 0.0), (1.0, 1.0)),
        lambda x: np.where(np.all(x == 0.5, axis=-1), value, 1.0),
        Dirichlet(lambda x: 0.0 * x[..., 0]),
    )


@pytest.mark.parametrize(
    ("network", "problem", "reason"),
    [
        (QuadraticForm(), T8, None),
        (QuadraticForm(), T20, "f(x0) <= 0"),  # f = 0 on a disk around (1/2, 1/2)
        (QuadraticForm(), T4, "f(x0) <= 0"),
        (QuadraticForm(), _f_at_the_centre(-1.0), "f(x0) <= 0"),
        (QuadraticForm(), _f_at_the_centre(np.inf), "f(x0) not finite"),
        (QuadraticForm(), _f_at_the_centre(np.nan), "f(x0) not finite"),
        (QuadraticForm(reference=(1.5, 0.0)), T8, "x0 outside"),
        (QuadraticForm(reference=(0.0, 0.0, 0.0)), T8, "reference dim"),
    ],
)
def test_the_refusal_says_which_condition_on_the_reference_fails(
    network, problem, reason
):
    assert network.refusal(problem) == reason
    assert network.supports(problem) == (reason is None)


def test_supports_needs_the_reference_inside_with_positive_f():
    assert QuadraticForm().supports(T8) and QuadraticForm().supports(T12)
    assert QuadraticForm().supports(T9)
    assert not QuadraticForm().supports(T20)  # f = 0 at (1/2, 1/2)
    assert QuadraticForm(reference=(0.1, 0.1)).supports(T20)
    assert QuadraticForm(reference=(0.1, 0.1)).supports(T4)
    assert not QuadraticForm(reference=(1.5, 0.0)).supports(T8)
    assert not QuadraticForm(reference=(1.0, 0.0)).supports(T8)  # on the boundary
    assert not QuadraticForm(reference=(0.0, 0.0, 0.0)).supports(T8)


def test_a_given_reference_is_where_the_form_is_centred():
    module = _module(QuadraticForm(reference=(0.25, -0.5)))
    assert module.report["reference"] == [0.25, -0.5]
    torch.testing.assert_close(
        module.reference, torch.tensor([0.25, -0.5], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"ratio": 0.0}, ValueError),
        ({"ratio": 1.0}, ValueError),
        ({"ratio": -0.5}, ValueError),
        ({"ratio": "big"}, TypeError),
        ({"width": 0}, ValueError),
        ({"depth": 1.5}, TypeError),
        ({"activation": "relu"}, ValueError),
        ({"reference": 3.0}, TypeError),
        ({"reference": "12"}, TypeError),
        ({"reference": b"12"}, TypeError),
        ({"reference": (0.0, True)}, TypeError),
        ({"reference": ()}, ValueError),
        ({"reference": (0.0, float("nan"))}, ValueError),
    ],
)
def test_the_part_rejects_bad_parameters(kwargs, error):
    with pytest.raises(error):
        QuadraticForm(**kwargs)


def test_the_reference_is_stored_as_floats():
    assert QuadraticForm(reference=[0, 1]).reference == (0.0, 1.0)


# The ICNN plus a quadratic


@pytest.mark.parametrize(("dim", "seed"), [(2, 0), (3, 1)])
def test_the_icnn_plus_a_quadratic_has_hessian_at_least_alpha(dim, seed):
    problem = {2: T1, 3: T9}[dim]
    module = _scramble(_module(ICNN(depth=3, alpha=0.4), problem=problem), seed=seed)
    hess = F.hessian(module, _points(500, dim, seed, scale=2.0), create_graph=False)
    eigmin = torch.linalg.eigvalsh(hess)[:, 0]
    assert (eigmin >= 0.4 - 1e-10 * hess.abs().amax()).all()


def test_the_icnn_quadratic_adds_no_parameter_and_draws_the_same_start():
    plain, plus = _module(ICNN(), problem=T1), _module(ICNN(alpha=0.5), problem=T1)
    for a, b in zip(plain.parameters(), plus.parameters(), strict=True):
        assert torch.equal(a, b)
    x = _points(30, 2, 5)
    with torch.no_grad():
        torch.testing.assert_close(plus(x) - plain(x), 0.25 * x.pow(2).sum(dim=-1))


def test_the_icnn_alpha_must_be_nonnegative():
    with pytest.raises(ValueError):
        ICNN(alpha=-0.1)
    assert ICNN().alpha == 0.0


# The presets


def test_the_dirichlet_preset_is_the_icnn_preset_with_the_qs_form():
    preset, icnn = DirichletQSPINN(), DirichletICNN()
    assert preset.network == QuadraticForm(width=10, depth=5, ratio=0.6)
    assert preset.reference is None
    assert (
        preset.loss
        == icnn.loss
        == PenaltyLoss(
            convexity_weight=0.0, boundary_weight=100.0, equation_at_boundary=True
        )
    )
    assert preset.sampler == icnn.sampler
    assert preset.optimizer == BFGS()
    assert (preset.max_steps, preset.tol) == (icnn.max_steps, icnn.tol)


def test_the_transport_preset_is_the_icnn_preset_with_the_qs_form():
    preset, icnn = TransportQSPINN(), TransportICNN()
    assert preset.network == QuadraticForm(width=10, depth=4, ratio=0.6)
    assert preset.loss == icnn.loss == HausdorffLoss(boundary_weight=1.0)
    assert preset.sampler == icnn.sampler == TransportSampler()
    assert preset.optimizer == Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200)
    assert (preset.max_steps, preset.tol) == (icnn.max_steps, icnn.tol)


@pytest.mark.parametrize(
    ("preset", "kwargs"),
    [
        (DirichletQSPINN, {"ratio": 1.5}),
        (DirichletQSPINN, {"boundary_weight": 0.0}),
        (DirichletQSPINN, {"boundary_points": 0}),
        (TransportQSPINN, {"ratio": 0.0}),
        (TransportQSPINN, {"adam_steps": -1}),
    ],
)
def test_the_presets_reject_bad_parameters(preset, kwargs):
    with pytest.raises(ValueError):
        preset(**kwargs)


def test_support_of_the_presets():
    dirichlet, transport = DirichletQSPINN(), TransportQSPINN()
    assert dirichlet.supports(T8) and dirichlet.supports(T9)
    assert not dirichlet.supports(T12) and not dirichlet.supports(T20)
    assert not dirichlet.supports(T4)
    assert transport.supports(T12) and not transport.supports(T8)
    with pytest.raises(UnsupportedProblemError, match=r"Transport boundary$"):
        dirichlet.solve(T12)  # not the reference point: the message is the usual one


@pytest.mark.parametrize("problem", [T4, T20])
def test_the_presets_name_the_reference_point_when_it_fails(problem):
    message = r"reference point x0 = \(0.5, 0.5\), f\(x0\) <= 0; pass reference="
    with pytest.raises(UnsupportedProblemError, match=message):
        DirichletQSPINN().solve(problem)
    moved = DirichletQSPINN(reference=(0.1, 0.1))
    assert moved.network == QuadraticForm(reference=(0.1, 0.1))
    assert moved.reference == (0.1, 0.1) and moved.supports(problem)


def test_the_transport_preset_names_the_reference_point_when_it_fails():
    with pytest.raises(UnsupportedProblemError, match="x0 outside"):
        TransportQSPINN(reference=(2.0, 0.0)).solve(T12)


def test_the_transport_preset_checks_the_mass():
    heavy = Problem(
        "heavy", Disk((0.0, 0.0), 1.0), lambda x: 2.0 + 0.0 * x[..., 0], T12.boundary
    )
    with pytest.raises(ValueError, match="mass"):
        TransportQSPINN(**SMALL, target_points=20).solve(heavy)


def test_the_qs_preset_trains_on_the_icnn_presets_points():
    qs = DirichletQSPINN(max_steps=3, seed=4, **SMALL).solve(T8)
    icnn = DirichletICNN(max_steps=1, seed=4, **SMALL).solve(T8)
    fingerprint = qs.info["collocation"]["fingerprint"]
    assert fingerprint == icnn.info["collocation"]["fingerprint"]
    assert qs.info["network"]["reference"] == [0.0, 0.0]
    assert qs.info["network"]["alpha_start"] == pytest.approx(2**-0.5, rel=1e-15)
    assert qs.info["network"]["alpha"] > 0  # the value after the training
    assert qs.info["unknowns"] == 518
    assert np.isfinite(evaluate(qs)["max"])  # convex only under the note's conditions


def test_the_transport_preset_trains_a_few_steps_on_t12():
    result = TransportQSPINN(
        max_steps=4, adam_steps=2, target_points=20, **SMALL
    ).solve(T12)
    assert result.info["iterations"] >= 1
    assert set(result.info["loss_terms"]) == {"loss", "equation", "into", "onto"}
    assert np.isfinite(evaluate(result)["max"])


def test_a_composed_qs_solver_rebuilds_from_its_hparams():
    pytest.importorskip("jsonargparse")
    from maesolvers import from_config

    solver = NeuralSolver(
        network=QuadraticForm(width=3, depth=2, ratio=0.25, reference=(0.1, -0.2)),
        loss=PenaltyLoss(convexity_weight=0.0, boundary_weight=10.0),
        sampler=UniformSampler(interior_points=10, boundary_points=8),
        optimizer=BFGS(),
        max_steps=2,
    )
    assert from_config(json.loads(json.dumps(solver.hparams))) == solver
    with_icnn = NeuralSolver(
        network=ICNN(alpha=0.25),
        loss=solver.loss,
        sampler=solver.sampler,
        optimizer=solver.optimizer,
    )
    assert from_config(with_icnn.hparams) == with_icnn


def test_a_problem_on_a_box_in_one_dimension_builds_a_scalar_form():
    line = Problem(
        "line",
        Box((0.0,), (1.0,)),
        lambda x: 1.0 + 0.0 * x[..., 0],
        Dirichlet(lambda x: 0.5 * (x[..., 0] - 0.5) ** 2),
    )
    module = _scramble(_module(QuadraticForm(width=3, depth=1), problem=line))
    x = torch.linspace(0, 1, 7, dtype=torch.float64)[:, None]
    assert module.matrix(x).shape == (7, 1, 1)
    assert module(x).shape == (7,)
