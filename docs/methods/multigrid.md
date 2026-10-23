# Multigrid

The full multigrid method of Liu, Froese, Oberman and Xiao (2017) solves the standard centred discretisation of `det D^2 u = f` on a box in two or three dimensions in a time that grows linearly with the number of unknowns.
It is a nonlinear full-approximation-storage (FAS) multigrid with full multigrid (FMG) initialisation, smoothed by a nonlinear Gauss-Seidel sweep that solves each node's equation for the root with a positive definite discrete Hessian, so it selects the convex solution where Newton's method may not.
maesolvers ships it as the iteration `Multigrid`, which composes with a scheme on a box (ADR 0004), the scheme `StandardBox` (the standard scheme on a 2D or 3D box), and the preset `FDMultigrid` for the pair.
Reproduction: [`benchmarks/multigrid.md`](../../benchmarks/multigrid.md).

```python
from maesolvers import T9, FDMultigrid, Multigrid, StandardBox
from maesolvers.metrics import evaluate

result = FDMultigrid(n=32).solve(T9)  # the preset
same = Multigrid(scheme=StandardBox(n=32), ordering="lexicographic").solve(T9)
print(evaluate(result)["max"], result.info["iterations"])  # 1.08e-4 after one FMG cycle
```

## The scheme

`StandardBox(n)` is the grid of `n` cells per side over a box (`h = 1/n` on the unit box, ADR 0001), the unknowns at the `(n - 1)^d` interior nodes and the Dirichlet data on the boundary ring.
At each interior node the discrete Hessian `D^2_h u` is the centred second differences of `maesolvers.classical.grid.second_differences`: `D_kk u = (u(i + e_k) - 2 u(i) + u(i - e_k)) / h_k^2` and the 4-point cross differences `D_kl`, 9 points in 2D and 19 in 3D (Liu et al. 2017, eqs. (4) to (9)).
The equation is

```text
det D^2_h u = f    at every interior node
```

(their eq. (10)); in 2D it is the equation of `Standard9` on a box, and a test checks that the two residuals and Jacobians agree.
`StandardBox` evaluates it by slicing the array of node values, so a residual costs a few passes over the nodes and no matrix; it builds the sparse Jacobian (`sum_k C_kk D_kk + 2 sum_{k<l} C_kl D_kl`, with the cofactors `C` of the discrete Hessian) and the factorisation of the Laplacian only when `Newton` asks for them.
So `Newton(scheme=StandardBox(n))` solves the same equations, and a grid of `8192^2` nodes fits in a few GB under the multigrid iteration.
The scheme is not monotone: like `Standard9` it has non-convex roots, and the iteration decides which root a solve reaches.

## The nonlinear Gauss-Seidel smoother

Hold every neighbour of a node fixed and the node's equation is a polynomial of degree `d` in its own value `v`.
With `A_kk = u(i + e_k) + u(i - e_k)` (the sum of the two axial neighbours) and `A_kl = h_k h_l D_kl u` (the cross differences, which do not involve `v`), the discrete Hessian is `H^-1 (A - 2 v I) H^-1` with `H = diag(h)`, so the equation reads

```text
det(A - t I) = f prod_k h_k^2,    t = 2 v.
```

`StandardBox.relax(grid, u, b, nodes)` sets `v` from its smallest real root (`maesolvers.classical.standard_box.smallest_root`): for `b > 0` that root lies below every eigenvalue of `A`, and it is the only root at which the node's discrete Hessian is positive definite (Liu et al. 2017, Theorems 2.1 and 2.2; in 2D, Benamou-Froese-Oberman 2010, which the paper takes over).
A fixed point of the sweeps is therefore a solution of the discrete equation with a positive definite discrete Hessian at every node, and every such solution is a fixed point.
In 2D the root is the closed form `(A_00 + A_11)/2 - sqrt(((A_00 - A_11)/2)^2 + A_01^2 + c)`; in 3D it is Newton's method on the cubic from a start below the root, from which the iterates rise to it monotonically (the cubic is decreasing and convex there).
Where `b <= 0`, which only the right-hand side of a coarse grid in a FAS cycle can be, the smallest real root lies at or above the smallest eigenvalue, and it is taken as the paper says ("the smallest of all the real roots", eq. (12)); in 2D, where there may be none, the vertex `(A_00 + A_11)/2` is taken.
Clamping `b` at 0 there instead stalls the cycles on T2 (`benchmarks/multigrid.md`, Finding 3).

A sweep visits the interior nodes in one of two orders, the parameter `ordering`:

- `"lexicographic"`, the paper's 3D order and its Table 2: node by node in flat order (the last axis fastest), each node reading its neighbours' newest values.
  `sweep_groups` reproduces it exactly in vectorised steps: the node with multi-index `i` goes into the wavefront `sum_k (d - k) i_k`; for the 9- and 19-point stencils every neighbour earlier in flat order lies in an earlier wavefront, every later one in a later wavefront, and none in the same, so a wavefront updated at once gives the sequential sweep to the last bit (a test compares the two).
- `"red-black"`, the paper's Table 1: the nodes with an even index sum, then the odd ones, each colour at once.
  The diagonal neighbours of a node have its colour, so for these stencils a colour's update is a Jacobi step among its nodes; this is what a vectorised red-black sweep of a 9-point stencil does, and what the paper's MATLAB code does by its account (Section 4.1, "faster vectorization").

## The multigrid cycle

The grids have `n`, `ceil(n/2)`, ... cells per side down to 2, the scheme rediscretised on each by `dataclasses.replace(scheme, n=...)` (the paper's coarse operator, "the discretized equations with a coarser step size").
For `n` a power of 2 the grids are nested, as in the paper; the transfers are its own: the residual goes down by half-weighting (`half_weighting`: the node `1/2`, each axial neighbour `1/(4 d)`, which is the paper's `[1 6 1]/12` stencil in 3D), the approximation by injection, and the correction comes up by multilinear interpolation.
Other `n` take the same operators through `resample`, the linear interpolation between two grids of the same box, which is injection onto a nested coarse grid and multilinear prolongation from one.

One V-cycle on a grid (Algorithm 1) from `w` for `S(w) = b`, with `S` the scheme's operator `det D^2_h`:

1. `pre_sweeps` sweeps (2);
2. the restricted residual `r_H = I(b - S(w))` and the injected `w_H`;
3. the coarse problem `S_H(u_H) = S_H(w_H) + r_H`, solved by the same cycle from `w_H`, on the coarsest grid by `coarsest_sweeps` sweeps (1);
4. `w += P(u_H - w_H)`, the interpolated correction;
5. `post_sweeps` sweeps (2).

The FMG cycle (Algorithm 2) starts on the coarsest grid, `coarsest_sweeps` sweeps from 0 inside, and on each finer grid interpolates the coarser solution by a cubic spline (`cubic_resample`: the paper says "cubic interpolation" and no more; ours is the tensor-product not-a-knot spline, of degree 2 from the 3-node grid), puts the Dirichlet data on the ring, and runs one V-cycle.
`Multigrid.initial_guess` does all of this below the finest grid, so that step 1 is the finest grid's V-cycle and completes the FMG cycle, and every later step is one more V-cycle; `Result.info["iterations"]` therefore counts as the paper's "Iter" does, 1 for the FMG cycle alone.

The stopping rule is the paper's: the discrete L2 norm of the residual `f - S(u)` at most `tol = 1e-6` times that of the reference, `Result.info["history"]` the ratio after each cycle.
The paper does not say what the reference of the FMG-FAS method is; ours is the residual of the node values that are 0 inside, with the Dirichlet data on the ring, the iterate from which the coarsest grid starts.
With it the iteration counts on T2 match Table 1, but the relative residuals after one cycle differ from the printed ones, mostly below them (`benchmarks/multigrid.md`, Finding 2).

## Source paper

J. Liu, B. D. Froese, A. M. Oberman, M. Xiao, "A multigrid scheme for 3D Monge-Ampere equations", Int. J. Comput. Math. 94 (2017) 1850-1866, doi 10.1080/00207160.2016.1247443: the scheme (Section 2, eqs. (4) to (12)), the FAS and FMG algorithms (Section 3, Algorithms 1 and 2) and the settings of the experiments (Section 4: 2 + 2 sweeps, one sweep on the grid `h_0 = 1/2`, the relative residual `1e-6`).
The 2D Gauss-Seidel update is that of J.-D. Benamou, B. D. Froese, A. M. Oberman, "Two numerical methods for the elliptic Monge-Ampere equation", M2AN 44 (2010) 737-758.

## Parameters

`StandardBox`:

| Parameter | Default | Meaning |
|---|---|---|
| `n` | 64 | cells per side, at least 2 |
| `spline_degree` | 3 | degree of the tensor spline through the node values (`Result.u`), 1 to 5, at most `n` |

`Multigrid` (and `FDMultigrid`, which takes `n` and `spline_degree` as well):

| Parameter | Default | Meaning |
|---|---|---|
| `scheme` | | a scheme on a box with a field `n`, unknowns at the interior nodes of `BoxGrid(problem.domain, n)` in flat order, and `residual` and `relax`, such as `StandardBox` |
| `ordering` | `"lexicographic"` | the order of the Gauss-Seidel sweeps, `"lexicographic"` or `"red-black"` |
| `pre_sweeps`, `post_sweeps` | 2, 2 | sweeps before and after the coarse-grid correction |
| `coarsest_sweeps` | 1 | sweeps on the grid of 2 cells per side |
| `max_iter` | 50 | most cycles, the FMG cycle included |
| `tol` | `1e-6` | the relative residual at which it stops |

A scheme of your own runs under `Multigrid` if it has these: `residual(grid, v)` (the multigrid operator is `residual + rhs`) and `relax(grid, u, b, nodes)`, which returns the new values at the flat node indices `nodes` from the values `u` at every node and the right-hand side `b` at those nodes, each node from the values it is given.

## Known limits

- A box with Dirichlet data in 2D or 3D only; `StandardBox` and `Multigrid` refuse other problems.
- The smoother selects the convex root node by node, but the scheme is not monotone, and the paper's convergence argument covers fixed points only: on degenerate data (`f = 0` on a set, T20, T4, T5) the paper reports that the method may break down, and nothing proves the cycles converge on singular solutions; on the paper's T2, T3 and T28 they do, on T22 up to `n = 64`.
- On the paper's 3D smooth examples one FMG cycle stops with a larger error than the paper prints, by up to a factor 2 (T9) or 1.6 (T46), though the paper's own Gauss-Seidel column, and ours, show the same discrete solution (`benchmarks/multigrid.md`, Finding 1).
- On T22 at `n = 128` the V-cycles do not settle on a discrete solution: the iterates repeat with a period of 15 (red-black) or 20 (lexicographic) cycles, with a residual of up to `1e5` at the nodes next to the singular corner (`benchmarks/multigrid.md`, Finding 4); the FMG cycle and the paper's stopping rule still stop within 25 percent of the paper's error.
- The lexicographic sweep takes `3 n` (2D) or `6 n` (3D) vectorised steps, so on small grids its fixed cost per step dominates and it is slower than red-black; on large ones both are linear in the unknowns.
- The paper sees rounding (a division by `h^6`) cost the second order on T9 at `n = 128` (its Section 4.2); in float64 ours keeps it there (the discrete solution's error falls by a factor 4.0 from `n = 64`), but the residual of `det D^2_h` carries that rounding, so a tolerance far below the paper's `1e-6` may not be reached on finer 3D grids.

## Reproduction

Tables 1 to 3 and 5 to 7 of the paper, with ours next to them, and Table 4's wide-stencil column against the 19-point monotone scheme: [`benchmarks/multigrid.md`](../../benchmarks/multigrid.md), written by `benchmarks/run_multigrid.py` on a100server (`multigrid.csv`).
