# Guide

This guide shows how to solve a problem, how to pass parameters to a solver, and how to write a solver of your own.
It assumes maesolvers is installed (see the README); the parts marked "needs the `cli` extra" also need `pip install "maesolvers[cli] @ git+https://github.com/jczhang02/maesolvers"`.
The Python blocks build on each other, as one session; `tests/test_guide.py` runs every code block below, so the code here works as written.

Three things make up the interface:

- a `Problem`: a domain, a right-hand side $f$, a boundary condition, and the exact solution when it is known;
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
    # T1 ['FDNewton', 'FDPoisson', 'FDMonotone', 'FDHybrid', 'FDFiltered', 'FDMixed', 'FDMultigrid', 'FDSemiLagrangian', ...], ...
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
It iterates $\Delta u_{k+1} = \sqrt{u_{xx}^2 + u_{yy}^2 + 2 u_{xy}^2 + 2 f}$ with the boundary data, from $\Delta u_0 = \sqrt{2 f}$, and stops when successive iterates differ by at most `tol` at every node.
Put the class in a module of your own, here `my_methods.py`.

```python
# my_methods.py
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from scipy.sparse.linalg import factorized

from maesolvers import Box, BoxGrid, Dirichlet, IterativeSolver, Solution
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
        grid = BoxGrid(problem.domain, self.n)  # (n + 1)^2 nodes, h = side / n
        inner = grid.interior  # the unknowns; the other nodes carry g
        dxx, dyy, dxy = difference_operators(self.n, *grid.h)
        laplace = dxx + dyy
        return SimpleNamespace(  # the context, passed to every later hook
            grid=grid,
            inner=inner,
            second_differences=(dxx, dyy, dxy),
            laplace=laplace,
            solve=factorized(laplace[:, inner].tocsc()),
            g=np.where(inner, 0.0, problem.boundary.g(grid.nodes)),
            f=problem.f(grid.nodes[inner]),
        )

    def poisson(self, context, rhs):
        """Solve laplace u = rhs at the interior nodes, with u = g on the boundary."""
        u = context.g.copy()
        u[context.inner] = context.solve(rhs - context.laplace @ context.g)
        return u

    def initial_guess(self, problem, context):
        return self.poisson(context, np.sqrt(2.0 * context.f))

    def step(self, problem, context, u):
        uxx, uyy, uxy = (d @ u for d in context.second_differences)
        rhs = np.sqrt(uxx**2 + uyy**2 + 2 * uxy**2 + 2 * context.f)
        new = self.poisson(context, rhs)
        return new, float(np.max(np.abs(new - u)))

    def converged(self, problem, context, measure):
        return measure <= self.tol

    def finalize(self, problem, context, u):
        return Solution(
            u=context.grid.interpolant(u, 3),  # the cubic tensor spline off the nodes
            nodes=context.grid.nodes,
            values=u,
            interior=context.inner,
            unknowns=int(context.inner.sum()),
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
  `BoxGrid(problem.domain, n)` is the grid of $n$ cells per side over a box that the shipped schemes use, in any dimension: its `nodes` in flat order, the mask `interior` of the nodes that carry unknowns (the others, the boundary ring, carry the Dirichlet data), the spacing $h$, and `interpolant(values, degree)`, the tensor spline through values at the nodes.
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
A scheme in HJB form, a maximum over policies of linear operators, runs under `PolicyIteration` if it provides `policy_system(grid, v)`, which returns the sparse matrix $A$ and the vector $F$ of the optimal policy at $v$ (so that $A v - F$ is the residual), and `solve_laplace(grid, rhs)`; `MixedHJB` in `maesolvers/classical/mixed.py` is the model (`docs/methods/chen_wan_lin2018.md`).

### Grids in any dimension

`Standard9`, `Hybrid`, `Filtered`, `MixedHJB`, `SemiLagrangian` and `SecondBoundary` are two-dimensional (`dims` is `(2,)`; `supports` is False for T9 and T10); `Monotone17` with `stencil=19` is the 3D monotone scheme (`FDMonotone(n=31, stencil=19).solve(T9)`, `docs/methods/monotone17.md`, "In three dimensions"), built on the grid below.
`StandardBox`, the standard scheme on a box, runs in 2D and 3D under `Newton` and `Multigrid` (`docs/methods/multigrid.md`); a scheme of your own on a box runs under `Multigrid` if it has a field `n` and provides `values` (the node values with its boundary data on the ring) and `relax` (the pointwise Gauss-Seidel solve) besides `residual`.
`BoxGrid` and the pure functions in `maesolvers.classical.grid` work on a box of any dimension, so a scheme in 3D or 4D needs only its own equations:

- `nodes`, shape $((n + 1)^d, d)$, in C order (the last axis fastest), so `values.reshape(grid.shape)` is the $d$-dimensional array; for even $n$ the nodes of the grid with $n/2$ cells are the nodes with even indices;
- `interior`, the mask of the $(n - 1)^d$ nodes that carry the unknowns, and `interior_index`, their multi-indices; the boundary ring `~interior` carries the Dirichlet data `g(grid.nodes[~grid.interior])`;
- `offset(direction)`, the step of the flat index to the neighbour along a direction of index offsets, and `reach(direction)`, the fraction of that step that stays in the box at each interior node, for the arms of a wide stencil (the pure function `reach(index, n, direction)` gives it at any nodes);
- `stencil(n, offsets, weights)`, any linear stencil of width $1$ as a sparse matrix; `second_differences(n, h)`, the centred second differences (9 points in 2D, 19 in 3D, 33 in 4D); `centred_hessian(grid, values)`, the discrete Hessian at the nodes in the shape `Result.node_hessian` takes;
- `interpolant(values, degree)`, the tensor spline through values at the nodes, which `evaluate` reads on the shared evaluation grid ($51^3$ points in 3D, $21^4$ in 4D; fitting and evaluating it there takes well under a second, `benchmarks/nd_grids.md`); it is defined on the box, and off the box the 2D spline clamps the points to the box while the spline in any other dimension extends its end polynomials.

A grid method with the exact values at its nodes has no error there, in any dimension:

```python
from maesolvers import T9, BoxGrid, Result
from maesolvers.classical.grid import centred_hessian
from maesolvers.metrics import evaluate

grid = BoxGrid(T9.domain, 16)  # (0, 1)^3, h = 1/16
values = T9.exact(grid.nodes)
exact = Result(
    T9,
    grid.interpolant(values, 3),
    nodes=grid.nodes,
    values=values,
    node_hessian=centred_hessian(grid, values),
    interior=grid.interior,
)
m = evaluate(exact)
assert m["max"] == 0.0 and m["nonconvex_fraction"] == 0.0
print(m["max_grid"])  # the spline's error on the 51^3 shared grid
```

## Neural solvers (needs the `neural` extra)

A neural solver represents the solution by a network and trains it to satisfy the equation at collocation points.
Its classes import with numpy alone, so they can be configured and listed anywhere; a solve needs torch and lightning, in the `neural` extra (`pip install "maesolvers[neural] @ git+https://github.com/jczhang02/maesolvers"`), and raises `ImportError` naming the extra without them.

### Train the preset

`PenaltyMLP` is the penalty MLP of Nystrom and Vestberg (2023), with the paper's defaults: $3000$ interior and $400$ boundary points, a $5 \times 10$ sigmoid network inside the ansatz $u = G + D N$, and up to $5000$ L-BFGS steps.
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
The `seed` fixes the collocation points and the initial network, so the same call gives the same result on the same machine with the same number of torch threads (`result.info["threads"]`); a reported run uses the seeds $0$ to $9$ and `maesolvers.metrics.summarize` aggregates them (ADR 0007, 0008).
The training runs on the CPU by default; `accelerator="gpu"` or `"auto"` (a GPU when there is one) moves it to a GPU, where it rounds differently, and for networks this small the CPU is often as fast.
`result.info["network"]` says how the fits of the ansatz's $G$ and $D$ ended: iterations, final gradient, whether they converged, and their misfit on the boundary.
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

`DirichletQSPINN` swaps one part of `DirichletICNN`: its network is the QS form of QS-PINN, `QuadraticForm()`, $u = (x - x_0)^\top A^\top A (x - x_0) + b \cdot x + c$ with $A$ from a plain network, which lies above its tangent plane at the reference point $x_0$ (`docs/methods/qs_pinn.md`).

```python
from maesolvers import DirichletQSPINN, QuadraticForm

qs = NeuralSolver(
    network=QuadraticForm(width=10, depth=5, ratio=0.6),
    loss=icnn.loss,
    sampler=icnn.sampler,
    optimizer=icnn.optimizer,
    max_steps=icnn.max_steps,
)
assert all(
    getattr(qs, name) == getattr(DirichletQSPINN(), name)
    for name in ("network", "loss", "sampler", "optimizer", "max_steps")
)
```

`QuadraticForm(reference=(0.1, 0.1))`, or `DirichletQSPINN(reference=(0.1, 0.1))`, puts $x_0$ elsewhere than the centre of the domain's bounding box, the default; the network needs $x_0$ inside the domain and $f(x_0) > 0$, so with the default it refuses T4, T20 and T20c, whose $f$ vanishes around the centre.
`ICNN(alpha=0.25)` is the input convex network plus $0.25 \lvert x \rvert^2 / 2$, whose Hessian is at least $0.25 I$.

The collocation points are drawn from the seed by the sampler, so every solver with an equal sampler and the same seed trains on the same points: `solver.collocation(problem)` returns them, and `result.info["collocation"]["fingerprint"]` identifies them.

### Write your own network, loss, sampler or optimizer

A part is a frozen dataclass, like a solver: it subclasses `Network`, `Loss`, `Sampler` or `Optimizer` from `maesolvers.neural`, carries its own `@dataclass(frozen=True, kw_only=True)` decorator, and its fields are its hyperparameters, which `Result.hparams` records and configurations set.
The four parts below are a network that adds a convex quadratic to any other network, the MLP, the QS form or one of your own (`ICNN(alpha=...)` is the same for the input convex network alone), a loss that penalises a negative Laplacian instead of negative diagonal entries, a sampler that adds boundary points near a singular corner (Nystrom and Vestberg do this for T3, Section 4.3), and an optimizer that takes a gradient step with momentum.
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
  `batch` holds the tensors `interior` and `f`, `boundary`, `g` (the Dirichlet data, None for another boundary condition) and `f_boundary` (the right-hand side at the boundary points, for a loss that also puts the equation there), on the training device during training, `target`, the target set of a transport problem (None for another boundary condition), and `target_boundary`, points on the target's boundary (None when the sampler drew none); `maesolvers.neural.functional` holds the kernels: the autograd `gradient` and `hessian`, `monge_ampere_residual`, `diagonal_convexity_penalty`, `mean_squared_misfit` and, for transport problems, `signed_distance` and `nearest_squared_distances`.
- A sampler's `sample(problem, rng)` returns a `Collocation` and must draw only from `rng`, so that the seed decides the points.
  A `Collocation` has a third field, `target_boundary`, empty by default: a sampler for a transport loss that compares point sets fills it with points on the target's boundary, as `TransportSampler` does, and the solver passes them on as `batch.target_boundary`.
- `supports(problem)` says which problems a part handles; the solver supports a problem when its network, loss and sampler all do.
  A loss supports the Dirichlet problems by default, because `batch.g` is Dirichlet data; `LaplacianPenaltyLoss` reads it and so keeps the default, and a loss for another boundary condition overrides `supports` and does not read `batch.g`.
  `UniformSampler` asks the domain for `BoundarySampling`, and `HardDirichlet` for `BoundaryDistance` as well, protocols of `maesolvers.core` that `Box`, `Disk` and `CutQuarterDisk` implement; `Ellipse` implements `BoundarySampling` only.
- `__post_init__` checks every parameter with `integer`, `positive`, `nonnegative` or `boolean`, and a part it takes with `check_part`, as the shipped parts do: `PlusQuadratic(network=MLP)`, the class for an instance, then fails at construction with a message that says to write `MLP(...)`.
- A loss term built on `functional.mean_squared_misfit` raises `ValueError` when its points are empty, so a boundary term cannot drop out because the sampler gave no boundary points.
- An optimizer's `build(parameters)` returns a torch optimizer of the trainable parameters whose `step(closure)` makes one update (one iteration, with its line search if it has one): the training loop counts its steps by the calls of `step`, so `max_steps` and the step count in the result and in MLflow mean one update each.
  `torch.optim.LBFGS` makes up to `max_iter` iterations per call, $20$ by default, and stops on its own tolerances; build it with `max_iter=1`, `tolerance_grad=0` and `tolerance_change=0`, as the shipped `LBFGS` part does, which is the model to copy for a line-search method.

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

## Transport problems

T12 and T16 carry the boundary condition `Transport(target)`: $\nabla u$ maps the domain onto the target set, the second boundary value problem of optimal transport, with a uniform target density $1$.
Its solution is unique up to a constant, so `evaluate` subtracts each side's mean before it takes the errors of $u$ (ADR 0007).
The papers on these problems report the error of the transport map $\nabla u$ instead (Benamou-Froese-Oberman 2014, Tables 1 and 2; Caboussat-Peruso 2025, Table 2), which `gradient_map_error` in `maesolvers.metrics` measures, outside `evaluate`.
`maesolvers.core` holds the geometry of the target sets `Box`, `Disk` and `Ellipse` as pure functions: `signed_distance` (positive outside the target, negative inside, as $H$ in Benamou-Froese-Oberman 2014), `support_function` ($H^\ast(n)$, the largest $y \cdot n$ over the target) and `closest_boundary_point` (the nearest boundary point and the outward unit normal there).

```python
import numpy as np

from maesolvers import T16
from maesolvers.core import check_mass_balance, signed_distance, support_function
from maesolvers.metrics import gradient_fd, shared_grid

target = T16.boundary.target  # the square (-1/2, 1/2)^2
check_mass_balance(T16)  # the integral of f over the domain is the target's area

points = shared_grid(T16.domain)
transport_map = gradient_fd(T16.exact, points)  # grad u by central differences
inside = signed_distance(target, transport_map) <= 1e-9  # up to rounding
assert np.all(inside)  # the map lands in the target
print(support_function(target, np.array([[1.0, 0.0], [1.0, 1.0]])))  # [0.5 1. ]
```

`gradient_map_error(gradient, exact)` returns `gradient_max` and `gradient_rms`, the largest and the root mean square Euclidean norm of the difference over the points, under names that do not clash with those of `evaluate`.
The RMS weighs every point by $1 / M$; the L2 norm over the domain is about $\sqrt{\operatorname{volume}(\text{domain})}$ times it, which differs from it where the domain's area is not $1$, as on T12's unit disk, so a benchmark states which of the two a paper's "L2 error" is.
Pass the method's own gradient where it has one, such as the differences of a grid scheme at its nodes, and `gradient_fd` of `result.u` otherwise.
`gradient_fd` evaluates $u$ a small step past each point along each axis, so $u$ must be the solution there: a problem's exact solution and a network are.
A classical result's `u` is an interpolant of its grid values, and past the domain it is not the solution: the 2D box spline is constant there, the tensor spline in another dimension extends its end polynomials, and the Clough-Tocher interpolant of a curved domain extrapolates to first order.
For such a `u`, take the interior points only, `points[domain.interior(points)]`, or pass the method's own gradient.

### Solve one

`FDTransport` solves a transport problem by the finite-difference method of Benamou-Froese-Oberman 2014 (`docs/methods/bfo2014.md`).
Its grid covers a square: the domain's bounding box, or `box` for a domain whose bounding box is not a square, with `f` zero outside the domain.
The paper measures the map by centred differences at the grid nodes; `gradient_fd` of `result.u` with the step $h$ gives them at the interior nodes, since the spline through the node values interpolates them.

```python
from maesolvers import T16, FDTransport
from maesolvers.metrics import evaluate, gradient_fd, gradient_map_error

result = FDTransport(n=32).solve(T16)
h = result.info["h"][0]
points = result.nodes[result.interior]
errors = gradient_map_error(
    gradient_fd(result.u, points, step=h), gradient_fd(T16.exact, points)
)
# gradient_max 0.0214, gradient_rms 0.0126; the paper prints 0.0220, 0.0127
print(errors)
print(evaluate(result)["max"])  # the error of u, after subtracting the means
```

The scheme needs only the target's support function, $H^\ast(n) = \sup_{y \in Y} y \cdot n$.
A target of a type the helpers do not know, such as a rotated ellipse, gets one through the method of the `SupportFunction` protocol.
`FDTransport` solves such a problem in one line.
Its `check` runs `check_mass_balance` where `can_check_mass_balance(problem)` is True, a `Box`, `Disk` or `Ellipse` domain and target; for a domain or target of your own `volume` or `integrate` cannot run, so it warns that the balance is unchecked and solves.
Check the balance yourself, as the comment below does, or read `result.info["u_anchor"]`, the constant $u(x_0)$ that absorbs an imbalance and is of the order of the discretisation error when the data balance.

```python
from dataclasses import dataclass

import numpy as np

from maesolvers import Box, Ellipse, Problem, Transport


@dataclass(frozen=True)
class RotatedEllipse:
    """The ellipse M B_1, M symmetric positive definite."""

    matrix: tuple = ((0.6, 0.2), (0.2, 0.8))
    dim = 2

    @property
    def bbox(self):
        r = np.linalg.norm(np.asarray(self.matrix), axis=1)
        return Box(tuple(-r), tuple(r))

    def _level(self, x):
        z = np.asarray(x) @ np.linalg.inv(np.asarray(self.matrix)).T
        return np.sum(z**2, axis=-1)

    def contains(self, x):
        return self._level(x) <= 1.0

    def interior(self, x):
        return self._level(x) < 1.0

    def support_function(self, n):  # SupportFunction: |M n|
        return np.linalg.norm(np.asarray(n) @ np.asarray(self.matrix).T, axis=-1)


# Uniform densities: f is the target's area, pi det M = 0.44 pi, over the
# domain's, pi 0.8 0.4 = 0.32 pi, so the masses balance.
problem = Problem(
    "ellipses",
    Ellipse((0.0, 0.0), (0.8, 0.4)),
    lambda x: np.full(np.shape(x)[:-1], 0.44 / 0.32),
    Transport(RotatedEllipse()),
)
solver = FDTransport(n=32, directions=64, box=Box((-1.0, -1.0), (1.0, 1.0)))
result = solver.solve(problem)  # UserWarning: the mass balance is unchecked
print(result.info["converged"], result.info["u_anchor"])
```

### Declare Transport support

A solver supports transport problems as it supports any other.
Written from scratch, it lists `Transport` in its `boundaries`; composed of parts, it supports a problem when its parts do, so a scheme or a loss for transport problems returns True from `supports` for them.
The base `Solver.check` does not check the mass balance.
A solver written from scratch, or a subclass of a preset, calls `check_mass_balance(problem)` in its own `check`, after `super().check(problem)`, so that a solve raises `ValueError` before `setup` for a problem whose source mass differs from the target's volume.
A composed solver (`NeuralSolver`, or a scheme with an iteration) has no hook yet through which a part adds a check (#28, #29), so its caller runs `check_mass_balance(problem)` before `solve`, as the example below does.
`check_mass_balance` integrates `f` over a `Box`, a `Disk` or an `Ellipse` and raises `TypeError` for another domain; a solver on another domain integrates `f` itself and passes the result as `mass`.
Its default `rtol` of $10^{-3}$ suits a smooth $f$; for an $f$ that jumps, which the quadrature resolves only to about the jump over the number of nodes ($10^{-2}$ at the default $64$), pass a larger `rtol`, more `points` or the `mass`.
A scheme reads the target from `problem.boundary.target`: the scheme of Benamou-Froese-Oberman 2014 writes the condition at a boundary node as the largest $\nabla u \cdot n - H^\ast(n)$ over a set of directions $n$, with $H^\ast(n) =$ `support_function(target, n)` computed once per solve.
A loss reads it from `batch.target`; this one penalises the squared signed distance of $\nabla u$ at the boundary points, on top of the equation's residual.

```python
from dataclasses import dataclass

import numpy as np

from maesolvers import ICNN, LBFGS, T1, T12, NeuralSolver, Transport, UniformSampler
from maesolvers.core import check_mass_balance, nonnegative
from maesolvers.metrics import evaluate, gradient_fd, gradient_map_error, shared_grid
from maesolvers.neural import Loss, functional


@dataclass(frozen=True, kw_only=True)
class TransportPenaltyLoss(Loss):
    """Mean squared residual plus the mean squared signed distance of grad u at the boundary."""

    transport_weight: float = 1.0

    def __post_init__(self):
        weight = nonnegative("transport_weight", self.transport_weight)
        object.__setattr__(self, "transport_weight", weight)

    def supports(self, problem):
        return isinstance(problem.boundary, Transport)

    def __call__(self, u, batch):
        hess = functional.hessian(u, batch.interior)
        image = functional.gradient(u, batch.boundary)  # where grad u maps the boundary
        terms = {
            "equation": functional.monge_ampere_residual(hess, batch.f).pow(2).mean(),
            "transport": functional.signed_distance(batch.target, image).pow(2).mean(),
        }
        total = terms["equation"] + self.transport_weight * terms["transport"]
        return {"loss": total, **terms}


transport = NeuralSolver(
    network=ICNN(width=10, depth=2),
    loss=TransportPenaltyLoss(),
    sampler=UniformSampler(interior_points=200, boundary_points=100),
    optimizer=LBFGS(),
    max_steps=20,
)
assert transport.supports(T12) and not transport.supports(T1)
check_mass_balance(T12)  # a composed solver does not check it itself
result = transport.solve(T12)
points = shared_grid(T12.domain)
print(gradient_map_error(gradient_fd(result.u, points), gradient_fd(T12.exact, points)))
print(evaluate(result)["max"])  # the error of u, after subtracting the means
```

`functional.signed_distance` takes its nearest boundary points from the numpy function, outside the autograd graph, so its gradient in $\nabla u$ is the outward normal there: the loss trains through it, with no NaN where $\nabla u$ lies on the target's boundary.

The shipped transport loss is `HausdorffLoss`, the loss of Caboussat-Peruso 2025: it compares the image $\nabla u$ of the boundary points with points on the target's boundary, both ways, so it needs no target geometry beyond those points, which `TransportSampler` draws.
The preset `TransportICNN` composes it with an input convex network fitted to $\nabla u = x$ before the training, and Adam then L-BFGS; as a preset subclass it checks the mass balance in its own `check`, and its `supports` takes only the domains and targets whose mass that check computes (`Box`, `Disk`, `Ellipse`).
`HausdorffLoss` supports every transport problem, but it needs both point sets: composed with a sampler that draws no target points, such as `UniformSampler`, the solver supports the problem and raises `ValueError` only at the first training step, after the network's `build` (for `IdentityStart`, after its fit to the identity map).

```python
from maesolvers import (
    HausdorffLoss,
    IdentityStart,
    Switch,
    TransportICNN,
    TransportSampler,
)
from maesolvers.neural import Adam

paper = NeuralSolver(
    network=IdentityStart(network=ICNN(width=10, depth=4)),
    loss=HausdorffLoss(boundary_weight=1.0),
    sampler=TransportSampler(
        interior_points=800, boundary_points=800, target_points=800
    ),
    optimizer=Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200),
    max_steps=2000,
    tol=1e-7,
)
assert all(
    getattr(paper, name) == getattr(TransportICNN(), name)
    for name in ("network", "loss", "sampler", "optimizer", "max_steps", "tol")
)
```

`TransportICNN(adam_steps=2000)` trains with Adam alone, the other row of the paper's Table 2; `docs/methods/icnn.md` gives the parameters and `benchmarks/icnn.md` the reproduction on T12.
