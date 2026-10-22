<div align="center">
  <h1>maesolvers</h1>
  <p><b>Solvers for the Monge-Amp&egrave;re equation, classical and neural, behind one interface and measured on one set of test problems.</b></p>
  <a href="https://github.com/jczhang02/maesolvers/actions/workflows/ci.yml"><img src="https://github.com/jczhang02/maesolvers/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/deps-numpy%20%2B%20scipy-lightgrey?style=flat-square" alt="numpy + scipy">
</div>

<p align="center">
  <img src="benchmarks/standard_fd.svg" alt="Max error against grid size for the standard finite-difference scheme: our Poisson iteration and Newton solver next to the values printed in Froese-Oberman 2011" width="900">
</p>

> maesolvers collects published methods for `det D^2 u = f`, reproduces the numbers their papers report, and runs them on a fixed catalogue of test problems so that methods can be compared on equal terms.
> It is pre-alpha (0.0.x): the interface may change until 0.1, and the first two solvers are in.
> The figure above is the first reproduction: the standard-scheme column of Froese-Oberman 2011, Table 3, matched to the printed digits, and a Newton iteration on the same scheme that converges to the wrong root on the singular problem.

## Features

- **One interface**: a `Problem` (domain, right-hand side, boundary condition, exact solution when known), any object with `solve(problem) -> Result`, and a `Result` that carries the solution as a callable plus the method's own nodes, values and discrete Hessian.
- **One catalogue**: test problems `P1` to `P5`, provisional until the research programme fixes its test set, with numbers that are never reused so that tables from different methods line up.
- **Shared metrics**: `evaluate(result)` reports solution errors at the method's nodes and on a common grid, the equation residual, convexity statistics and cost, the same way for every method.
- **Reproduction with receipts**: `benchmarks/` holds the published tables, our runs, the comparison and the script that draws it; a mismatch is recorded as a finding, not tuned away.
- **Light dependencies**: numpy and scipy only; torch arrives with the `neural` extra and the neural solvers.

## Quick Start

maesolvers is not on PyPI yet; install it from GitHub.

```sh
uv add git+https://github.com/jczhang02/maesolvers        # in a uv project
pip install git+https://github.com/jczhang02/maesolvers   # or with pip
```

Solve a catalogue problem and evaluate the result with the shared protocol.

```python
from maesolvers import P4, FDNewton
from maesolvers.metrics import evaluate

result = FDNewton(n=64).solve(P4)
print(evaluate(result))
```

```text
rel_l2_nodes: 6.31e-06       # relative L2 error at the method's own nodes
max_nodes: 1.68e-05          # max error at the nodes (the number papers print)
rel_l2_grid: 6.39e-06        # the same two errors on the shared 201 x 201 grid
max_grid: 1.68e-05
residual_rms: 3.12e-12       # det D^2 u_h - f, RMS and max, with the method's own Hessian
residual_max: 2.07e-11
eigmin_min: 1                # smallest Hessian eigenvalue over the nodes
nonconvex_fraction: 0        # share of nodes where it is negative
time: 0.0809                 # seconds
iterations: 4
unknowns: 3969
```

Any object with a `solve(problem)` method that returns a `Result` plugs into the same `evaluate`, so a new method is one class away from the comparison table.

## Test problems

All five are two-dimensional.
The catalogue is provisional: the research repository has not fixed its test problems yet, and the numbering will be aligned with it when it does.

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

## Reproduction

A method counts as reproduced when it matches the source paper's own table, on the paper's own test problem, to the printed digits.
`benchmarks/sources.md` transcribes the published tables with their conventions; `benchmarks/standard_fd.md` compares them with ours.

| Problem | N | Published | FDPoisson | FDNewton |
|---|---|---|---|---|
| P4 | 31 | 7.14e-5 | 7.144e-5 | 7.144e-5 |
| P4 | 361 | 0.05e-5 | | 5.276e-7 |
| P5 | 31 | 17.15e-3 | 17.15e-3 | 17.45e-3 |
| P5 | 361 | 5.41e-3 | 5.409e-3 | 21.94e-3 |

Two things came out of this first comparison.
The papers' grid convention, which none of them states, is `h = 1/N`.
Newton on the standard scheme converges in the residual on P5 but to a non-convex discrete solution near the corner, so the literature's "standard scheme" numbers are the fixed point of the Poisson or Gauss-Seidel iteration, not of Newton.

<details>
<summary><strong>Redraw the figure</strong></summary>

```sh
uv run benchmarks/plot_standard_fd.py     # reads the CSVs next to it, writes standard_fd.svg
```

</details>

## Roadmap

A method enters the library when it can be implemented from its paper and fits `Problem -> Result`; methods without an error table are included and marked as such.
The order puts first the methods the literature uses most, those that share infrastructure, and those with tables to reproduce.
The design behind 0.1 is recorded in `docs/adr/`.

- **0.0.2**: the solver lifecycle, composition and configuration of ADR 0003 to 0005, the frozen `Problem`, `Result` and `evaluate` of ADR 0007, and the catalogue under the T ids of the shared test set; the two existing solvers move over with their reproduction intact.
- **0.1**, finite differences: Shortley-Weller boundaries for the standard 9-point scheme (so T8 runs on the disk), the monotone 17-point scheme (Froese-Oberman 2011), the filtered scheme (Froese-Oberman 2013), the hybrid scheme (Froese-Oberman 2011); neural: the shared training loop (ADR 0008) with the penalty MLP (Nystrom 2023) and ICNN; the comparison table generated from the runs.
- **0.2**: n-dimensional boxes and the 19-point monotone scheme, transport boundary conditions (Benamou-Froese-Oberman 2014, Caboussat-Peruso 2025), a multigrid option, the semi-Lagrangian scheme, QS-PINN.
- **0.3**: finite elements through scikit-fem (Feng-Neilan, Brenner-Neilan C0 interior penalty, Awanou splines, two-scale, Dean-Glowinski least-squares and augmented Lagrangian).
- **0.4**: Oliker-Prussner, MA-LBR, Froese 2017 point clouds, C-PINN, KAN, Deep Ritz splitting.

Out of scope, because `Problem -> Result` cannot hold them: semi-discrete and sample-based optimal transport, parabolic formulations.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff format . && uv run ruff check .
```

Tests check that every catalogue problem's exact solution satisfies its own equation, that the solvers converge at the expected order, and that the published numbers above still match.

## License

MIT; see [LICENSE](LICENSE).
