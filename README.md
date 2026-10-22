# maesolvers

Solvers for the Monge-Ampere equation `det D^2 u = f`, classical and neural, behind one interface.
The library collects published methods, reproduces their reported numbers, and runs them on one shared set of test problems so that methods can be compared on equal terms.

Status: pre-alpha, version 0.0.x.
The interface may change until 0.1.

## Three concepts

- `Problem`: domain, right-hand side `f`, boundary condition (Dirichlet data `g`, or a transport target set), and the exact solution when known.
- A solver: any object with `solve(problem) -> Result`.
- `Result`: the solution as a callable, plus the method's own nodes, values and discrete Hessian.
  `maesolvers.metrics.evaluate(result)` applies the shared protocol: solution errors at the nodes and on a fixed grid, residual, convexity statistics, cost.

```python
from maesolvers import P4, FDNewton
from maesolvers.metrics import evaluate

result = FDNewton(n=64).solve(P4)
print(evaluate(result))
```

## Test problems

The catalogue `P1` to `P5` is fixed: numbers are never reused, a retired problem keeps its number.
All five are two-dimensional.

| Name | Domain | `f` | Boundary | Exact solution | Tests |
|---|---|---|---|---|---|
| P1 | unit disk | 1 | Dirichlet, `g = 1/2` | `|x|^2 / 2` | branch selection: the concave `1 - |x|^2 / 2` also solves the equation |
| P2 | `(-1, 1)^2` | `(1 + |x|^2)^-2` | Dirichlet | `sqrt(1 + |x|^2)` | smooth, non-quadratic |
| P3 | unit disk | 1 | transport onto the ellipse centred at `(3.5, 0)` with semi-axes `(2, 0.5)` | `x1^2 + 3.5 x1 + x2^2 / 4` | transport boundary condition (Caboussat-Peruso 2025) |
| P4 | `(0, 1)^2` | `(1 + |x|^2) exp(|x|^2)` | Dirichlet | `exp(|x|^2 / 2)` | smooth, the standard finite-difference example |
| P5 | `(0, 1)^2` | `2 (2 - |x|^2)^-2` | Dirichlet | `-sqrt(2 - |x|^2)` | gradient blow-up at the corner `(1, 1)` |

## Solvers

| Solver | Method | Domains | Boundary | Convexity |
|---|---|---|---|---|
| `FDNewton` | standard 9-point finite differences, damped Newton from a Poisson start | `Box` | Dirichlet | none; on P5 it converges to a non-convex discrete root |
| `FDPoisson` | the same discretisation solved by the Poisson iteration (M2) of Benamou-Froese-Oberman 2010 | `Box` | Dirichlet | selects the convex branch in practice |

Both reproduce the standard-scheme column of Table 3 in Froese-Oberman (JCP 2011, SIAM 2011) to three digits once the grid convention is `h = 1/N`; see `benchmarks/standard_fd.md`.

Planned for 0.1: one monotone wide-stencil scheme, one neural baseline (ICNN), Shortley-Weller boundary handling for disks.

## Reproduction rule

A method counts as reproduced when it matches the source paper's own table, on the paper's own test problem, to the same order of magnitude.
`benchmarks/sources.md` records the published numbers, `benchmarks/<method>.md` the comparison with ours.
A mismatch is recorded as a finding, not tuned away.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff format . && uv run ruff check .
```

Dependencies are numpy and scipy only; the `neural` extra adds torch for the neural solvers.
