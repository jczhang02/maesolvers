# Standard 9-point scheme

The standard 9-point scheme is the plainest finite-difference discretisation of `det D^2 u = f` in two dimensions.
maesolvers ships it as the scheme `Standard9` under two iterations, `PoissonIteration` and `Newton`, and as the presets `FDPoisson` and `FDNewton` for these two pairs (ADR 0004).
The two iterations solve the same discrete equations but can reach different discrete solutions, so they count as two methods.
Reproduction: [`benchmarks/standard_fd.md`](../../benchmarks/standard_fd.md); the curved-boundary checks: [`benchmarks/shortley_weller.md`](../../benchmarks/shortley_weller.md).

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

Off the nodes, the solution is a tensor spline of degree `spline_degree` through the node values (degree `n` on a grid with fewer cells per side; `Result.info["interpolant"]` names it); `Result.node_hessian` holds `D_xx`, `D_yy` and `D_xy` at the interior nodes and NaN at the boundary nodes.
The difference operators are the pure functions `difference_operators`, `monge_ampere` and `monge_ampere_jacobian` in `maesolvers.classical.schemes`.
On a box the grid is `BoxGrid(problem.domain, n)` and `difference_operators` is the 2D case of `second_differences` in `maesolvers.classical.grid`, which builds the centred differences on a box of any dimension.

## Curved boundaries (Shortley-Weller)

On a `Disk`, an `Ellipse` or the `CutQuarterDisk` of T6 the grid covers the domain's bounding box with `n` cells per side, so `h = 2/n` on the unit disk of T8 and `h = 1/n` on a domain whose bounding box is the unit square.
The nodes in the open domain carry the unknowns; the nodes outside it are dropped.
At a node `P` the scheme takes a second difference along each of four grid directions `v`: the axes `(hx, 0)`, `(0, hy)` and the diagonals `(hx, hy)`, `(hx, -hy)`.
When the neighbour `P + v` (or `P - v`) is in the open domain the difference uses it; otherwise it uses the point `P + a v` (or `P - b v`), `0 < a, b <= 1`, where the segment towards that neighbour leaves the domain, and the Dirichlet data `g` there.
Through the three values at `-b v`, `0` and `a v` the difference is the second derivative of the interpolating parabola (Shortley-Weller 1938):

```text
D_v u = 2 / (a b (a + b)) * (b u(P + a v) - (a + b) u(P) + a u(P - b v))
u_xx = D_(1,0) u / hx^2
u_yy = D_(0,1) u / hy^2
u_xy = (D_(1,1) u - D_(1,-1) u) / (4 hx hy)
```

With `a = b = 1` these are the nine-point stencils above, so away from the boundary nothing changes, and on a `Box` the code path is the one of the section above, bit for bit.
Next to the boundary the differences are first-order consistent but exact for quadratics, and the error at the nodes stays second order (`benchmarks/shortley_weller.md`).

Six details make the stencils robust:

- The crossings are found by bisection on `domain.interior`, so the scheme needs only the `Domain` protocol; `g` is evaluated at the end of the bisection bracket, a point on the boundary to rounding.
- A node of the open domain closer than `1e-6` steps to the boundary along a grid direction joins the boundary and takes the data `g` at itself.
  Rounding puts nodes that lie on the boundary in exact arithmetic, such as `(15/17, 8/17)` on the unit circle at `n = 34`, on either side of it, and such a node would otherwise get weights of order `1e15 / h^2`.
- A node outside the closed domain closer than `1e-6` steps to it along a grid direction joins the boundary too, such as `(0.6, 0.8)` on the unit circle at `n = 10`; each of its neighbours in the open domain would otherwise add a copy of it as a crossing.
  Points closer than about `1e-5` steps can remain (T6 at `n = 70` and `100`); Qhull then leaves one of them out of the triangulation, which is harmless: the interpolant reads gradients only at vertices of the hull, and the point left out is not one.
- A node on an edge of the bounding box joins the boundary if the closed domain contains it, even when rounding puts it in the open domain: the box only touches the domain there, and such a node would otherwise have no neighbour on one side.
- The weights grow like `1 / t`, `t` the smallest of the node's step fractions `a` and `b`, and the rounding error of its equation grows with them.
  `Standard9.residual` therefore returns `t ((D_xx u)(D_yy u) - (D_xy u)^2 - f)`, with `t = 1` on a box and at every node whose stencil is full.
  Scaling an equation changes neither the discrete solution nor a Newton direction; it keeps Newton's residual tolerance within reach of rounding (without it, Newton stalls unconverged on the ellipse and on T6 at `n = 128`, Finding 2 of `benchmarks/shortley_weller.md`), at the price of a tolerance looser by `1 / t` at the cut nodes.
  `Result.info["residual_max"]` reports this scaled residual, and `Result.info["min_step"]` the smallest `t`.
- Off the nodes, the solution is the piecewise cubic Clough-Tocher interpolant (scipy) through the unknowns and the boundary points, and `spline_degree` is not used (`Result.info["interpolant"]` is `"clough_tocher"`).
  The domain reaches past the convex hull of these points: along a smooth boundary in slivers of width `O(h^2)` between consecutive boundary points, and at a corner that is not a grid node, such as the two ends of T6's straight cut, in a wedge of width `O(h)`.
  A point `q` there is extrapolated to first order from its projection `p` on the nearest hull edge, `u(p) + grad u(p) . (q - p)`, with `u` linear along the edge and `grad u` linear between the gradients the Clough-Tocher interpolant estimates at the edge's ends; the error stays `O(h^2)` in the wedges as in the slivers.

`Result.nodes` lists the nodes with unknowns, then the boundary points, with `Result.interior` True at the first; `Result.node_hessian` is NaN at the boundary points.
The pure function `shortley_weller_operators(domain, n)` in `maesolvers.classical.schemes` returns them as the named tuple `ShortleyWellerOperators`: the nodes, the mask, the three operators and the smallest step fraction `t` of each stencil.
`shortley_weller_nodes(domain, n)` returns the nodes and the mask alone, which `SemiLagrangian` triangulates (`docs/methods/feng_jensen2017.md`).

On T8 the stencils are exact for the solution `(|x|^2 - 1)/2`, so the convex root of the discrete equation is the exact solution at the nodes, and so is its mirror image, the concave root `(1 - |x|^2)/2`.
`FDNewton` starts from `laplace u = 2 sqrt(f) = 2`, which here is already the convex root, and stops after one step; from `laplace u = -2` it reaches the concave root.
`FDPoisson` cannot reach the concave root: every iterate solves `laplace u = (a non-negative right-hand side)` with `g = 0`, so by the discrete maximum principle of the Shortley-Weller Laplacian it is `<= 0`, while the concave root is positive inside.

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
- G. H. Shortley, R. Weller, "The numerical solution of Laplace's equation", J. Appl. Phys. 9 (1938) 334-348, doi 10.1063/1.1710426: the unequal-spacing differences at the boundary, there for Laplace's equation; it prints no table for the Monge-Ampere equation, so the curved-boundary results are sanity checks, not a reproduction.

`benchmarks/sources.md` transcribes the tables and their conventions.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `Standard9` | `n` | 64 | cells per side of the grid over the bounding box, at least 2 |
| `Standard9` | `spline_degree` | 3 | degree of the spline off the nodes on a box, 1 to 5, at most `n` in effect; not used on a curved domain |
| `PoissonIteration` | `max_iter` | 10000 | most steps |
| `PoissonIteration` | `tol` | 1e-12 | largest change at a node between two iterates |
| `Newton` | `max_iter` | 50 | most steps |
| `Newton` | `tol` | 1e-10 | largest residual, relative to `max(1, max \|f\|)` |

The presets take the scheme's parameters flat: `FDPoisson(n=128, tol=1e-10)` is `PoissonIteration(scheme=Standard9(n=128), tol=1e-10)`, and `FDNewton(n=128)` is `Newton(scheme=Standard9(n=128))`.

## Known limits

- Only 2D Dirichlet problems on a `Box`, `Disk`, `Ellipse` or `CutQuarterDisk`: `solve` raises `UnsupportedProblemError` for a transport boundary (T12) or a box in 3D or 4D (T9, T10).
  Which schemes support which dimensions: `Standard9`, `Hybrid` and `Filtered` are 2D only (`dims = (2,)`); `Monotone17` with `stencil=19` is the 3D monotone scheme (`docs/methods/monotone17.md`, "In three dimensions").
  The grid under them (`BoxGrid`: nodes, boundary ring, Dirichlet data, the centred Hessian and the tensor spline onto the shared evaluation grid) works in any dimension (issue #25, `docs/guide.md`, "Grids in any dimension"); a 3D standard scheme, the determinant of a 3 by 3 Hessian and its Jacobian, is not written, and comes with multigrid (#30).
- On a curved domain the crossings are looked for only on the segment from a node to an outside neighbour, so a boundary that crosses one grid segment twice (a non-convex domain with features finer than the grid) is not resolved; the domains `Standard9` declares are convex.
- On a curved domain the interpolant's error on the shared grid (`max_grid`) falls at the rate of the node error, corners off the grid included, but is 7 to 10 times it on a disk and on T6 and up to 18 times on a stretched grid, against about the same on a box with the cubic spline: the Clough-Tocher interpolant estimates its gradients to `O(h)` only (`benchmarks/shortley_weller.md`).
- `FDNewton` reaches a non-convex discrete solution on T2 at every `n` and on T7 from `n = 40` on: its residual converges, but its error does not decrease with `n` (Finding 2 of `benchmarks/standard_fd.md`).
  Report `FDPoisson`, not `FDNewton`, as the literature's "standard finite differences".
  On T2, which non-convex root Newton reaches depends on rounding (Finding 4).
- Newton's default `tol = 1e-10`, relative to `max f`, meets the rounding floor of the nine-point stencil near `n = 512`: on T1 it still converges on the box, by 10 percent, but not on the inscribed disk, whose `max f` is half as large (Finding 3 of `benchmarks/shortley_weller.md`); pass a larger `tol` there.
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
