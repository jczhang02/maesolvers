# Guide

This guide shows how to solve a problem, how to pass parameters to a solver, and how to write a solver of your own.
It assumes maesolvers is installed (see the README); the parts marked "needs the `cli` extra" also need `pip install "maesolvers[cli] @ git+https://github.com/jczhang02/maesolvers"`.
The Python blocks build on each other, as one session; `tests/test_guide.py` runs every code block below, so the code here works as written.

Three things make up the interface:

- a `Problem`: a domain, a right-hand side `f`, a boundary condition, and the exact solution when it is known;
- a solver, whose `solve(problem)` returns a `Result`: the solution as a callable `u(points)`, and for a grid method its nodes, its values there and its discrete Hessian;
- `maesolvers.metrics.evaluate(result)`, which measures any result the same way.

## Solve a problem

Every test problem is in the catalogue under its T id.
Import a problem by its id, or look it up in `CATALOGUE`.

```python
from maesolvers import CATALOGUE, T1, FDPoisson
from maesolvers.metrics import evaluate

assert CATALOGUE["T1"] is T1
print(T1.domain, type(T1.boundary).__name__, T1.note)

result = FDPoisson(n=64).solve(T1)
error = evaluate(result)["max"]  # the max error at the nodes
assert f"{error:.2e}" == "1.68e-05"
```

A result carries what the solver computed and how the solve went.

```python
print(result.u([[0.5, 0.5], [0.25, 0.75]]))  # the solution anywhere in the domain
assert result.nodes.shape == (65 * 65, 2) and result.values.shape == (65 * 65,)
print(result.info["iterations"], result.info["converged"], result.info["time"])
print(result.hparams)  # the configuration that rebuilds the solver
```

`evaluate` returns a dict.
The errors `max` and `rms` are taken at the method's own nodes, or on the shared evaluation grid for a method without nodes; `max_grid` and `rms_grid` repeat them on the shared grid through the interpolant, a secondary value.
`nonconvex_fraction` and `eigmin` measure convexity from the solver's Hessian, and the problem's observables (for T7, `u_origin`) come last.

A solver declares which problems it supports, so a loop over problems and solvers asks first instead of catching errors.
`SOLVERS` holds the presets by name and `SHARED_TEST_SET` the ids of the comparison table's rows.

```python
from maesolvers import SHARED_TEST_SET, SOLVERS

for name in SHARED_TEST_SET:
    problem = CATALOGUE[name]
    supported = [key for key, preset in SOLVERS.items() if preset().supports(problem)]
    print(name, supported)
    # T1 ['FDNewton', 'FDPoisson', 'FDMonotone', 'FDHybrid', 'FDFiltered', 'PenaltyMLP', 'DirichletICNN'], ...
```

Solving an unsupported problem raises `UnsupportedProblemError`, before any work is done.
`benchmarks/run_comparison.py` runs this loop with every preset's defaults to make the README comparison table.

A callback observes a solve without changing it.
Subclass `Callback` and override the hooks you need: `on_solve_start`, `on_step_end` and `on_solve_end`.

```python
from maesolvers import T2, Callback


class EveryTenth(Callback):
    def on_step_end(self, solver, step, metrics):
        if step % 10 == 0:
            print(step, metrics)  # {"update": ...} for the Poisson iteration


FDPoisson(n=16).solve(T2, callbacks=[EveryTenth()])
```

`maesolvers.tracking.MLflowLogger` is a callback that logs a solve into MLflow; it needs mlflow, which is not a dependency of the library.

## Pass parameters

A solver's parameters are the fields of its class, and there are three ways to set them.
All three read the same constructor signatures, so they accept the same names and values.

### Keyword arguments

```python
from maesolvers import FDNewton, Newton, Standard9

solver = FDNewton(n=128, tol=1e-12)
composed = Newton(scheme=Standard9(n=128), tol=1e-12)  # the same method, from its parts
```

`FDNewton` is a preset: it names the common pair of the standard 9-point scheme and Newton's method.
Every solver takes keyword arguments only and checks them on construction: `FDNewton(n=1)` raises `ValueError`, and `FDNewton(n=1.5)` raises `TypeError`.

### A configuration (needs the `cli` extra)

A configuration names the solver by its `class_path` with its `init_args`, nested parts in the same form.
It is a dict, or a YAML or JSON file with the same content.

```yaml
# newton.yaml
class_path: Newton
init_args:
  scheme:
    class_path: Standard9
    init_args: {n: 64}
  tol: 1.0e-12
```

```python
from maesolvers import from_config

solver = from_config("newton.yaml")
assert solver == Newton(scheme=Standard9(n=64), tol=1e-12)

solver = from_config({"class_path": "FDPoisson", "init_args": {"n": 64}})
result = solver.solve(T1)
rerun = from_config(result.hparams).solve(result.problem)  # any result reruns
```

A misspelt key or a value of the wrong type raises `ValueError` with the key's name.

### The command line (needs the `cli` extra)

`maesolvers solve` takes the solver as `--solver` with dotted overrides, and the problem by its T id.
A configuration file holds the same keys: `solver`, `problem` and, optionally, `output`.

```yaml
# exp.yaml
solver:
  class_path: Newton
  init_args:
    scheme:
      class_path: Standard9
      init_args: {n: 64}
problem: T2
```

```sh
maesolvers solve --config exp.yaml
maesolvers solve --config exp.yaml --solver.scheme.n 128 --problem T1 --output t1.json
maesolvers solve --solver FDPoisson --solver.n 32 --problem T20
```

An argument after `--config` overrides the file.
The command prints `evaluate` as a table, and `--output` writes the metrics, `Result.info` and a configuration that reruns the result, as JSON.
`maesolvers solve --solver.help Newton` lists the parameters of a solver, and `--print_config` writes a configuration that `--config` reads back.

## Write your own solver

There are three plug-in points, from the smallest to the most general:

- a new iteration for the existing schemes: subclass `SchemeIteration`, or an iteration that already exists;
- a new finite-difference scheme for the existing iterations: subclass `Scheme`;
- any other method: subclass `IterativeSolver`, or `Solver` for a method that is not a loop of steps.

Whichever you pick, the base class runs the fixed lifecycle `check -> setup -> run -> finalize`: it times the solve, calls the callbacks, records the hyperparameters and builds the `Result`.
So `evaluate`, configurations and the command line work for your solver as they do for the solvers maesolvers ships, and it needs no registration.

### A solver from scratch

The worked example is the Poisson iteration of Benamou, Froese and Oberman (2010) on the standard 9-point scheme, written against `IterativeSolver` alone, so that its result can be checked against `FDPoisson`.
It iterates `laplace u_{k+1} = sqrt(u_xx^2 + u_yy^2 + 2 u_xy^2 + 2 f)` with the boundary data, from `laplace u_0 = sqrt(2 f)`, and stops when successive iterates differ by at most `tol` at every node.
Put the class in a module of your own, here `my_methods.py`.

```python
# my_methods.py
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import factorized

from maesolvers import Box, Dirichlet, IterativeSolver, Solution
from maesolvers.classical.schemes import difference_operators
from maesolvers.core import integer, positive


@dataclass(frozen=True, kw_only=True)
class MyPoisson(IterativeSolver):
    """The Poisson iteration of Benamou, Froese and Oberman (2010) on a 2D box."""

    domains: ClassVar = (Box,)
    boundaries: ClassVar = (Dirichlet,)
    dims: ClassVar = (2,)
    measure: ClassVar = "update"  # the name of what step returns

    n: int = 32  # cells per side
    tol: float = 1e-10
    max_iter: int = 1000

    def __post_init__(self):
        super().__post_init__()  # checks the class, and that max_iter is an integer >= 1
        object.__setattr__(self, "n", integer("n", self.n, 2))
        object.__setattr__(self, "tol", positive("tol", self.tol))

    def setup(self, problem):
        (x0, y0), (x1, y1) = problem.domain.lo, problem.domain.hi
        x, y = np.linspace(x0, x1, self.n + 1), np.linspace(y0, y1, self.n + 1)
        nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
        inner = np.zeros((self.n + 1, self.n + 1), dtype=bool)
        inner[1:-1, 1:-1] = True
        inner = inner.ravel()
        dxx, dyy, dxy = difference_operators(self.n, x[1] - x[0], y[1] - y[0])
        laplace = dxx + dyy
        return SimpleNamespace(  # the context, passed to every later hook
            x=x,
            y=y,
            nodes=nodes,
            inner=inner,
            second_differences=(dxx, dyy, dxy),
            laplace=laplace,
            solve=factorized(laplace[:, inner].tocsc()),
            g=np.where(inner, 0.0, problem.boundary.g(nodes)),
            f=problem.f(nodes[inner]),
        )

    def poisson(self, grid, rhs):
        """Solve laplace u = rhs at the interior nodes, with u = g on the boundary."""
        u = grid.g.copy()
        u[grid.inner] = grid.solve(rhs - grid.laplace @ grid.g)
        return u

    def initial_guess(self, problem, grid):
        return self.poisson(grid, np.sqrt(2.0 * grid.f))

    def step(self, problem, grid, u):
        uxx, uyy, uxy = (d @ u for d in grid.second_differences)
        new = self.poisson(grid, np.sqrt(uxx**2 + uyy**2 + 2 * uxy**2 + 2 * grid.f))
        return new, float(np.max(np.abs(new - u)))

    def converged(self, problem, grid, measure):
        return measure <= self.tol

    def finalize(self, problem, grid, u):
        spline = RectBivariateSpline(grid.x, grid.y, u.reshape(self.n + 1, -1))

        def u_at(points):
            p = np.asarray(points, dtype=float)
            values = spline(p[..., 0].ravel(), p[..., 1].ravel(), grid=False)
            return values.reshape(p.shape[:-1])

        return Solution(
            u=u_at,
            nodes=grid.nodes,
            values=u,
            interior=grid.inner,
            unknowns=int(grid.inner.sum()),
        )
```

What each part does:

- `@dataclass(frozen=True, kw_only=True)` on the class itself: the fields are the hyperparameters that `Result.hparams` records, and a solver is immutable, so a solve keeps its working data in the context, never on the solver.
  An inherited decorator does not count; without its own, or with a constructor parameter that can be passed by position or is not a field (an `InitVar`), construction raises `TypeError`.
  A subclass that adds no field may omit `kw_only=True`, but not the decorator.
- `domains`, `boundaries` and `dims` declare the supported problems; `solve` raises `UnsupportedProblemError` for any other.
- `measure` names what `step` returns; callbacks receive it as `{"update": ...}` and `Result.info["history"]` holds it for every step.
- `__post_init__` checks the type and the range of every parameter, after `super().__post_init__()`, so that a wrong value fails on construction.
  `integer` and `positive` from `maesolvers.core` are the checks the solvers maesolvers ships use: they raise `TypeError` or `ValueError` with a message that names the parameter, and return a plain `int` or `float`, which `object.__setattr__` stores on the frozen solver, so that `hparams` holds no numpy scalars.
- `setup` builds what the solve needs once; every later hook takes the problem and this context first.
- `step` returns the next iterate and its measure, or None when it can make no progress; it must not change the iterate it was given.
- `finalize` turns the final iterate into a `Solution`, the method's part of the result.
  `u` must accept points of shape `(..., 2)`, and `nodes`, `values` and `interior` let `evaluate` report the errors at the nodes; add `node_hessian` or `hessian` for the convexity fields.
  Its `info` adds entries to `Result.info`; `"stop"` says why the run ended, and a run that blew up sets `"stop": "diverged"`, which `summarize` counts as a failed seed whatever its error.

Use it like any solver.

```python
import numpy as np
from my_methods import MyPoisson

mine = MyPoisson(n=32).solve(T1)
print(evaluate(mine))
print(mine.info["iterations"], mine.info["converged"])

reference = FDPoisson(n=32, tol=1e-10).solve(T1)
np.testing.assert_allclose(mine.values, reference.values, atol=1e-12)
```

With the `cli` extra, a configuration or the command line names it by its dotted `class_path`.
`python -m maesolvers` puts the working directory on the import path, so it finds `my_methods.py` there; the `maesolvers` command needs the directory on `PYTHONPATH`.

```python
solver = from_config({"class_path": "my_methods.MyPoisson", "init_args": {"n": 16}})
assert from_config(solver.hparams) == solver
```

```sh
python -m maesolvers solve --solver my_methods.MyPoisson --solver.n 16 --problem T1
```

### A new iteration

An iteration solves the discrete equations of any scheme that has the methods it names in `requires`.
Subclass `SchemeIteration`, or an existing iteration to change one hook.
This one under-relaxes the Poisson iteration.

```python
from dataclasses import dataclass

import numpy as np

from maesolvers import PoissonIteration
from maesolvers.core import positive


@dataclass(frozen=True, kw_only=True)
class RelaxedPoisson(PoissonIteration):
    """The Poisson iteration with the update scaled by omega."""

    omega: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        omega = positive("omega", self.omega)
        if omega > 1.0:
            raise ValueError(f"omega must be at most 1, got {omega}")
        object.__setattr__(self, "omega", omega)

    def step(self, problem, grid, iterate):
        new, _ = super().step(problem, grid, iterate)
        relaxed = iterate + self.omega * (new - iterate)
        return relaxed, float(np.max(np.abs(relaxed - iterate)))


relaxed = RelaxedPoisson(scheme=Standard9(n=32), omega=0.8).solve(T1)
print(relaxed.info["iterations"], evaluate(relaxed)["max"])
```

`requires` is inherited from `PoissonIteration`: `second_derivatives` and `solve_laplace`, which `Standard9` has.
A new iteration that calls other scheme methods lists them in its own `requires`, and `solve` then raises `TypeError` for a scheme without them.

### A new scheme

A scheme subclasses `Scheme`, carries its own `@dataclass(frozen=True, kw_only=True)` decorator, so its fields are its parameters (an iteration checks this as the base class checks a solver), and implements `supports`, `discretize`, `rhs` and `solution`, plus the methods the iterations it serves require.
`Standard9` in `maesolvers/classical/schemes.py` is the model to follow; `docs/methods/standard9.md` describes its mathematics.

## Neural solvers (needs the `neural` extra)

A neural solver represents the solution by a network and trains it to satisfy the equation at collocation points.
Its classes import with numpy alone, so they can be configured and listed anywhere; a solve needs torch and lightning, in the `neural` extra (`pip install "maesolvers[neural] @ git+https://github.com/jczhang02/maesolvers"`), and raises `ImportError` naming the extra without them.

### Train the preset

`PenaltyMLP` is the penalty MLP of Nystrom and Vestberg (2023), with the paper's defaults: 3000 interior and 400 boundary points, a 5 x 10 sigmoid network inside the ansatz `u = G + D N`, and up to 5000 L-BFGS steps.
Here it trains briefly on fewer points.

```python
from maesolvers import T6, PenaltyMLP

solver = PenaltyMLP(interior_points=500, boundary_points=100, max_steps=50, seed=0)
result = solver.solve(T6)
print(evaluate(result))  # on the shared evaluation grid: the network has no nodes
print(result.info["stop"], result.info["iterations"], result.info["loss_terms"])
print(result.u([[0.2, 0.3]]), result.hessian([[0.2, 0.3]]))  # autograd Hessian
```

The result is measured like any other: `evaluate` takes the errors on the shared evaluation grid, never at the training points, and the convexity fields from the autograd Hessian there.
`result.info["stop"]` says why the training ended: `"converged"` (the largest gradient component fell to `tol`), `"max_steps"`, `"stalled"` (a step changed nothing) or `"diverged"` (the loss was not finite).
The `seed` fixes the collocation points and the initial network, so the same call gives the same result on the same machine with the same number of torch threads (`result.info["threads"]`); a reported run uses the seeds 0 to 9 and `maesolvers.metrics.summarize` aggregates them (ADR 0007, 0008).
The training runs on the CPU by default; `accelerator="gpu"` or `"auto"` (a GPU when there is one) moves it to a GPU, where it rounds differently, and for networks this small the CPU is often as fast.
`result.info["network"]` says how the fits of the ansatz's `G` and `D` ended: iterations, final gradient, whether they converged, and their misfit on the boundary.
Every step reaches the callbacks with the loss terms as metrics, so `MLflowLogger` logs a training curve.

### Compose the parts

A neural solver is four parts (ADR 0004): a network, a loss, a sampler and an optimizer, run by the shared training loop of `NeuralSolver`.
`PenaltyMLP()` is this composition:

```python
from maesolvers import (
    LBFGS,
    MLP,
    HardDirichlet,
    NeuralSolver,
    PenaltyLoss,
    UniformSampler,
)

composed = NeuralSolver(
    network=HardDirichlet(network=MLP(width=10, depth=5, activation="sigmoid")),
    loss=PenaltyLoss(convexity_weight=1e4),
    sampler=UniformSampler(interior_points=3000, boundary_points=400),
    optimizer=LBFGS(),
    max_steps=5000,
)
assert all(
    getattr(composed, name) == getattr(PenaltyMLP(), name)
    for name in ("network", "loss", "sampler", "optimizer", "max_steps")
)
```

Swap one part to get another method: `MLP()` as the network with `PenaltyLoss(boundary_weight=100.0)` imposes the boundary condition by a penalty instead of the ansatz, and `Adam(lr=1e-3)` replaces L-BFGS.
The second preset, `DirichletICNN`, swaps three parts: the input convex network `ICNN()`, convex in its input by construction, a loss with a boundary penalty instead of the convexity term, and `BFGS()`, which keeps a dense inverse Hessian approximation.

```python
from maesolvers import BFGS, ICNN, DirichletICNN

icnn = NeuralSolver(
    network=ICNN(width=10, depth=5),
    loss=PenaltyLoss(
        convexity_weight=0.0, boundary_weight=100.0, equation_at_boundary=True
    ),
    sampler=UniformSampler(interior_points=3000, boundary_points=400),
    optimizer=BFGS(),
    max_steps=10_000,
)
assert all(
    getattr(icnn, name) == getattr(DirichletICNN(), name)
    for name in ("network", "loss", "sampler", "optimizer", "max_steps")
)
```

The collocation points are drawn from the seed by the sampler, so every solver with an equal sampler and the same seed trains on the same points: `solver.collocation(problem)` returns them, and `result.info["collocation"]["fingerprint"]` identifies them.

### Write your own network, loss, sampler or optimizer

A part is a frozen dataclass, like a solver: it subclasses `Network`, `Loss`, `Sampler` or `Optimizer` from `maesolvers.neural`, carries its own `@dataclass(frozen=True, kw_only=True)` decorator, and its fields are its hyperparameters, which `Result.hparams` records and configurations set.
The four parts below are a network that adds a convex quadratic to any other network, a loss that penalises a negative Laplacian instead of negative diagonal entries, a sampler that adds boundary points near a singular corner (Nystrom and Vestberg do this for T3, Section 4.3), and an optimizer that takes a gradient step with momentum.
Put them in a module of your own, here `my_parts.py`.

```python
# my_parts.py
from dataclasses import dataclass

import numpy as np
import torch

from maesolvers.core import check_part, integer, nonnegative, positive
from maesolvers.neural import (
    Collocation,
    Loss,
    Network,
    Optimizer,
    UniformSampler,
    functional,
)


class _PlusQuadratic(torch.nn.Module):
    def __init__(self, inner, alpha):
        super().__init__()
        self.inner, self.alpha = inner, alpha

    def forward(self, x):
        return self.inner(x) + 0.5 * self.alpha * (x**2).sum(dim=-1)


@dataclass(frozen=True, kw_only=True)
class PlusQuadratic(Network):
    """Another network plus alpha |x|^2 / 2."""

    network: Network
    alpha: float = 1.0

    def __post_init__(self):
        check_part("PlusQuadratic.network", self.network, Network)
        object.__setattr__(self, "alpha", nonnegative("alpha", self.alpha))

    def supports(self, problem):
        return self.network.supports(problem)

    def build(self, problem, collocation):
        return _PlusQuadratic(self.network.build(problem, collocation), self.alpha)


@dataclass(frozen=True, kw_only=True)
class LaplacianPenaltyLoss(Loss):
    """Mean squared residual, a penalty on min(0, laplace u), and the boundary misfit."""

    convexity_weight: float = 100.0
    boundary_weight: float = 100.0

    def __post_init__(self):
        for name in ("convexity_weight", "boundary_weight"):
            object.__setattr__(self, name, nonnegative(name, getattr(self, name)))

    def __call__(self, u, batch):
        hess = functional.hessian(u, batch.interior)
        laplacian = torch.diagonal(hess, dim1=-2, dim2=-1).sum(dim=-1)
        terms = {
            "equation": functional.monge_ampere_residual(hess, batch.f).pow(2).mean(),
            "convexity": torch.clamp(laplacian, max=0.0).pow(2).mean(),
            "boundary": functional.mean_squared_misfit(u(batch.boundary), batch.g),
        }
        loss = (
            terms["equation"]
            + self.convexity_weight * terms["convexity"]
            + self.boundary_weight * terms["boundary"]
        )
        return {"loss": loss, **terms}


@dataclass(frozen=True, kw_only=True)
class CornerSampler(UniformSampler):
    """Uniform points, plus `extra` boundary points within `radius` of `corner`."""

    extra: int = 40
    corner: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "extra", integer("extra", self.extra, 0))
        object.__setattr__(self, "radius", positive("radius", self.radius))

    def sample(self, problem, rng):
        points = super().sample(problem, rng)
        near = np.empty((0, problem.dim))
        while len(near) < self.extra:
            draw = problem.domain.sample_boundary(10 * self.extra, rng)
            close = np.linalg.norm(draw - self.corner, axis=-1) < self.radius
            near = np.concatenate([near, draw[close]])
        boundary = np.concatenate([points.boundary, near[: self.extra]])
        return Collocation(interior=points.interior, boundary=boundary)


@dataclass(frozen=True, kw_only=True)
class Momentum(Optimizer):
    """Gradient descent with momentum, one update per step."""

    lr: float = 1e-2
    momentum: float = 0.9

    def __post_init__(self):
        object.__setattr__(self, "lr", positive("lr", self.lr))
        momentum = nonnegative("momentum", self.momentum)
        if momentum >= 1.0:
            raise ValueError(f"momentum must be below 1, got {momentum}")
        object.__setattr__(self, "momentum", momentum)

    def build(self, parameters):
        return torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum)
```

What each part does:

- A network's `build(problem, collocation)` returns a torch module that maps points of shape `(M, d)` to values of shape `(M,)`.
  The solver calls it once per solve, with float64 as torch's default dtype and torch's CPU generator seeded from the solver's seed, so a module made there is float64 and reproducible; the parameters that require a gradient are the ones the optimizer trains.
  A network that fits something before training, like `HardDirichlet`, does it in `build`, where the collocation points are at hand, and may set an attribute `report` on the module, a dict of plain values, which the solver records as `result.info["network"]`.
- A loss's `__call__(u, batch)` returns a dict of scalar tensors, with the total under `"loss"`; every entry is logged after each step under its name.
  `batch` holds the tensors `interior` and `f`, `boundary`, `g` (the Dirichlet data, None for another boundary condition) and `f_boundary` (the right-hand side at the boundary points, for a loss that also puts the equation there), on the training device during training, and `maesolvers.neural.functional` holds the kernels: the autograd `hessian`, `monge_ampere_residual`, `diagonal_convexity_penalty` and `mean_squared_misfit`.
- A sampler's `sample(problem, rng)` returns a `Collocation` and must draw only from `rng`, so that the seed decides the points.
- `supports(problem)` says which problems a part handles; the solver supports a problem when its network, loss and sampler all do.
  A loss supports the Dirichlet problems by default, because `batch.g` is Dirichlet data; `LaplacianPenaltyLoss` reads it and so keeps the default, and a loss for another boundary condition overrides `supports` and does not read `batch.g`.
  `UniformSampler` asks the domain for `BoundarySampling`, and `HardDirichlet` for `BoundaryDistance` as well, protocols of `maesolvers.core` that `Box`, `Disk` and `CutQuarterDisk` implement.
- `__post_init__` checks every parameter with `integer`, `positive`, `nonnegative` or `boolean`, and a part it takes with `check_part`, as the shipped parts do: `PlusQuadratic(network=MLP)`, the class for an instance, then fails at construction with a message that says to write `MLP(...)`.
- A loss term built on `functional.mean_squared_misfit` raises `ValueError` when its points are empty, so a boundary term cannot drop out because the sampler gave no boundary points.
- An optimizer's `build(parameters)` returns a torch optimizer of the trainable parameters; the training loop calls its `step(closure)` once per training step, so a line-search method such as `torch.optim.LBFGS` runs one iteration per step.

Use them like the shipped parts.

```python
from my_parts import CornerSampler, LaplacianPenaltyLoss, Momentum, PlusQuadratic

from maesolvers import T3

mine = NeuralSolver(
    network=PlusQuadratic(network=MLP(width=8, depth=2, activation="softplus")),
    loss=LaplacianPenaltyLoss(),
    sampler=CornerSampler(interior_points=300, boundary_points=80),
    optimizer=Momentum(lr=1e-3),
    max_steps=30,
    accelerator="cpu",
)
result = mine.solve(T3)  # f = 1/|x| is singular at the corner (0, 0)
print(result.info["loss_terms"], evaluate(result)["max"])
assert len(mine.collocation(T3).boundary) == 80 + 40
```

With the `cli` extra, a configuration names a part of your own by its dotted `class_path`, nested in the solver's configuration, and `Result.hparams` records it the same way.

```python
from my_parts import PlusQuadratic

config = {
    "class_path": "NeuralSolver",
    "init_args": {
        "network": {
            "class_path": "my_parts.PlusQuadratic",
            "init_args": {"network": {"class_path": "MLP"}, "alpha": 0.5},
        },
        "loss": {"class_path": "my_parts.LaplacianPenaltyLoss"},
        "sampler": {"class_path": "my_parts.CornerSampler", "init_args": {"extra": 20}},
        "optimizer": {"class_path": "my_parts.Momentum", "init_args": {"lr": 1e-3}},
    },
}
assert from_config(config).network == PlusQuadratic(network=MLP(), alpha=0.5)
assert from_config(mine.hparams) == mine
```
