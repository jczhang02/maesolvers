# Standard 9-point scheme

The standard 9-point scheme is the plainest finite-difference discretisation of `det D^2 u = f` in two dimensions.
maesolvers ships it as the scheme `Standard9` under two iterations, `PoissonIteration` and `Newton`, and as the presets `FDPoisson` and `FDNewton` for these two pairs (ADR 0004).
The two iterations solve the same discrete equations but can reach different discrete solutions, so they count as two methods.
Reproduction: [`benchmarks/standard_fd.md`](../../benchmarks/standard_fd.md).

## The scheme

The domain is a box `[x0, x1] x [y0, y1]` with `n` cells per side, so the spacing is `hx = (x1 - x0)/n` and `hy = (y1 - y0)/n` (`h = 1/n` on the unit square, ADR 0001), and the grid has `(n + 1)^2` nodes.
The unknowns are the values at the `(n - 1)^2` interior nodes; the boundary nodes carry the Dirichlet data `g`.
At an interior node the second derivatives are the centred differences

```text
D_xx u = (u[i+1, j] - 2 u[i, j] + u[i-1, j]) / hx^2
D_yy u = (u[i, j+1] - 2 u[i, j] + u[i, j-1]) / hy^2
D_xy u = (u[i+1, j+1] - u[i+1, j-1] - u[i-1, j+1] + u[i-1, j-1]) / (4 hx hy)
```

nine points together, and the discrete equation is

```text
(D_xx u)(D_yy u) - (D_xy u)^2 = f    at every interior node.
```

The scheme is consistent and second order for a smooth solution.
It is not monotone, and its equation does not see the sign of the Hessian: a concave or saddle-shaped grid function can satisfy it as well as the convex one.
Which root a solve reaches is decided by the iteration (branch selection).

Off the nodes, the solution is a tensor spline of degree `spline_degree` through the node values; `Result.node_hessian` holds `D_xx`, `D_yy` and `D_xy` at the interior nodes and NaN at the boundary nodes.
The difference operators are the pure functions `difference_operators`, `monge_ampere` and `monge_ampere_jacobian` in `maesolvers.classical.schemes`.

## The Poisson iteration

The Poisson iteration (M2) of Benamou, Froese and Oberman (2010) solves one Poisson problem per step:

```text
laplace u_0     = sqrt(2 f)
laplace u_{k+1} = sqrt((D_xx u_k)^2 + (D_yy u_k)^2 + 2 (D_xy u_k)^2 + 2 f)
```

with the Dirichlet data at the boundary nodes, where `laplace` is `D_xx + D_yy`, factorised once by sparse LU.
For a convex solution the right-hand side equals its Laplacian, since `(u_xx + u_yy)^2 = u_xx^2 + u_yy^2 + 2 u_xy^2 + 2 f`, and it is never negative, so the iteration selects the convex branch in practice (Froese-Oberman 2011, Section 8.1).
It stops when two successive iterates differ by at most `tol` at every node; the measure in `Result.info["history"]` is that difference (`update`).
The number of steps grows with the singularity of the problem: about 50 on T1 at every `n`, about `n^2` on T7.

## Newton's method

`Newton` takes damped Newton steps on the residual `(D_xx u)(D_yy u) - (D_xy u)^2 - f` with its sparse Jacobian `diag(D_yy u) D_xx + diag(D_xx u) D_yy - 2 diag(D_xy u) D_xy`.
It starts from `laplace u_0 = 2 sqrt(f)`, a lower bound for the Laplacian of the convex solution: `u_xx + u_yy >= 2 sqrt(f)` by the inequality of arithmetic and geometric means on the Hessian's eigenvalues.
A step of length `t` is halved until it cuts the max residual by the factor `1 - 1e-4 t`; when 40 halvings do not, the solve stops unconverged.
It stops when the max residual is at most `tol * max(1, max |f|)`; the measure is that residual (`residual`).
On T1 it needs 3 steps at `n = 8` and 16, and 4 at every `n` tried from 24 to 361.
Nothing in the iteration prefers the convex root, so on a singular problem it may converge to a non-convex one.

## Source papers

- J.-D. Benamou, B. D. Froese, A. M. Oberman, "Two numerical methods for the elliptic Monge-Ampere equation", ESAIM: M2AN 44 (2010) 737-758, doi 10.1051/m2an/2010017: the scheme and the Poisson iteration (M2); Table 7 is reproduced on T7.
- B. D. Froese, A. M. Oberman, "Fast finite difference solvers for singular solutions of the elliptic Monge-Ampere equation", J. Comput. Phys. 230 (2011) 818-834, doi 10.1016/j.jcp.2010.10.020: the "Standard" column of Table 3, reproduced on T1 and T2, and the report (Section 8.1) that Newton on this scheme fails on T2.

`benchmarks/sources.md` transcribes the tables and their conventions.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `Standard9` | `n` | 64 | cells per side, at least 2 |
| `Standard9` | `spline_degree` | 3 | degree of the spline off the nodes, 1 to `min(5, n)` |
| `PoissonIteration` | `max_iter` | 10000 | most steps |
| `PoissonIteration` | `tol` | 1e-12 | largest change at a node between two iterates |
| `Newton` | `max_iter` | 50 | most steps |
| `Newton` | `tol` | 1e-10 | largest residual, relative to `max(1, max \|f\|)` |

The presets take the scheme's parameters flat: `FDPoisson(n=128, tol=1e-10)` is `PoissonIteration(scheme=Standard9(n=128), tol=1e-10)`, and `FDNewton(n=128)` is `Newton(scheme=Standard9(n=128))`.

## Known limits

- Only a 2D `Box` with a `Dirichlet` boundary: `solve` raises `UnsupportedProblemError` for a disk (T8), a transport boundary (T12) or a box in 3D or 4D (T9, T10).
  Shortley-Weller boundaries for curved domains are planned for 0.1.
- `FDNewton` reaches a non-convex discrete solution on T2 at every `n` and on T7 from `n = 40` on: its residual converges, but its error does not decrease with `n` (Finding 2 of `benchmarks/standard_fd.md`).
  Report `FDPoisson`, not `FDNewton`, as the literature's "standard finite differences".
  On T2, which non-convex root Newton reaches depends on rounding (Finding 4).
- The Poisson iteration converges slowly on problems whose solution is singular at the boundary: T7 at `n = 140` needs about 15000 steps, more than the default `max_iter`, so pass `max_iter=100_000` there.
- On T2 the error converges at about order `h^0.5`, against `h^2` on T1, and the spline overshoots near the corner, so `max_grid` exceeds the node error at coarse `n`.

## Reproduction

At the levels of ADR 0002, as `benchmarks/standard_fd.md` states them:

| Problem | Source | `FDPoisson` | `FDNewton` |
|---|---|---|---|
| T1 | Froese-Oberman 2011, Table 3 | exact at `N = 31` to 127, not run beyond | exact, against the column of the Poisson and Gauss-Seidel iterations, whose discrete solution it reaches |
| T2 | Froese-Oberman 2011, Table 3 | exact | mismatch (Finding 2) |
| T7 | Benamou-Froese-Oberman 2010, Table 7 | exact | mismatch from `N = 41` (Finding 2) |

`tests/test_reproduction.py` pins the published numbers; the full comparison and the findings are in [`benchmarks/standard_fd.md`](../../benchmarks/standard_fd.md).
