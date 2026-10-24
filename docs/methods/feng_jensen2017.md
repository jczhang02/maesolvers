# Semi-Lagrangian scheme

The semi-Lagrangian scheme of Feng and Jensen (2017) writes `det D^2 u = f` as a Hamilton-Jacobi-Bellman (HJB) equation, whose solutions are convex without a convexity constraint, and discretises it with wide stencils whose ends fall anywhere in the domain and take the value of the piecewise linear interpolant there.
The scheme is monotone, its discrete solution is unique, and Howard's algorithm finds it from any start.
maesolvers ships it as the scheme `SemiLagrangian`, the iteration `Howard`, and the preset `FDSemiLagrangian` for that pair (ADR 0004).
Reproduction: [`benchmarks/feng_jensen2017.md`](../../benchmarks/feng_jensen2017.md).

```python
from maesolvers import T39, FDSemiLagrangian, Howard, SemiLagrangian
from maesolvers.metrics import evaluate

result = FDSemiLagrangian(n=80, stencil_factor=4).solve(T39)  # the preset
same = Howard(scheme=SemiLagrangian(n=80, stencil_factor=4)).solve(T39)  # its parts
print(evaluate(result)["max"] / 4)  # 7.6e-3, relative to max |u| = 4
```

## The equation in HJB form

For a symmetric matrix `A` and `f >= 0` (Feng-Jensen 2017, Section 3; the paper writes the right-hand side as `(f/d)^d`, so its `f` is `2 sqrt(f)` here):

```text
H(A, f) = sup over B >= 0, tr B = 1 of  -B : A + 2 sqrt(f) sqrt(det B).
```

`H(A, f) = 0` holds exactly when `A` is positive semidefinite and `det A = f` (their Lemma 3.2), and the viscosity solutions of `H(D^2 u, f) = 0` are the convex viscosity solutions of `det D^2 u = f` (Theorems 3.3 and 3.5).
Convexity comes from the operator: `-B : D^2 u <= 0` for every `B` forces `D^2 u >= 0`.

Write `B = a s1 s1^T + (1 - a) s2 s2^T` for an orthonormal frame `(s1, s2)` and `0 <= a <= 1`.
Then `B : D^2 u = a d1 + (1 - a) d2`, with `d1`, `d2` the second derivatives along the frame, and the supremum over `a` has a closed form (Cauchy-Schwarz, with `a = (1 + s) / 2`):

```text
sup over a of  -(a d1 + (1 - a) d2) + 2 sqrt(f) sqrt(a (1 - a))  =  sqrt(((d1 - d2) / 2)^2 + f) - (d1 + d2) / 2.
```

It is at most zero exactly when `d1 d2 >= f` and `d1 + d2 >= 0`, the condition of the monotone scheme of Froese and Oberman with the convexity built in (`frame_values`).
The maximising split is `a = (1 - (d1 - d2) / (2 r)) / 2`, `r` the square root above (`frame_weights`); where `r = 0` every split attains it and `a = 1/2`.

## The scheme

The frames are those of `directions` unit vectors at the angles `i pi / directions`, `0 <= i < directions` (`stencil_directions`); with an even count, direction `i` and direction `i + directions / 2` form a frame.
The paper's 64 rotation angles with `a < 1/2`, plus `B = Id / 2`, are the 32 frames of the 64 directions with `a` in `[0, 1]`.

Along each direction `s` the second derivative at a node `x` is the semi-Lagrangian difference

```text
d_s u(x) = (I u(x + k s) - 2 u(x) + I u(x - k s)) / k^2,
```

with `I u` the continuous piecewise linear interpolant of the nodal values (`semi_lagrangian_operators`, `linear_weights`).
The interpolation weights are nonnegative, so the scheme is monotone.
At every unknown the discrete equation is

```text
max over frames of  sqrt(((d1 - d2) / 2)^2 + f) - (d1 + d2) / 2  =  0,
```

and at a root `d1 d2 >= f`, `d1 >= 0`, `d2 >= 0` in every frame: the discrete solution is convex along every stencil direction.

### Nodes and stencil size

The paper works on a quasi-uniform unstructured triangulation.
`SemiLagrangian` uses the nodes of the finite-difference family instead: a uniform grid with `n` cells per side over the domain's bounding box (`h = 1/n` on the unit square, ADR 0001), the grid nodes in the open domain carrying the unknowns, and as boundary points the grid nodes on the boundary and the points where grid lines and diagonals cross it, as `shortley_weller_nodes` places them; the Dirichlet data are imposed there.
The method `place_nodes(domain)` returns them, so a subclass that overrides it puts the scheme on other nodes, as long as their triangulation spans the domain and the nodes it marks as unknowns lie strictly inside its convex hull (`discretize` raises `ValueError` otherwise).
The interpolant lives on the Delaunay triangulation of these nodes, whose union is a polygon `Omega_h` inscribed in the domain; every supported domain is convex, so `Omega_h` is their convex hull.

The stencil size is `k = stencil_factor * h` (the paper's `k = m h`, with `h` the larger spacing), reduced near the boundary to the distance from the node to the boundary of `Omega_h`, so that both ends stay in the triangulation (the paper's Remark 4.1 and Section 8).
Near the boundary `k` can fall below `h`; there the difference is no longer consistent, which the paper's analysis allows (its consistency holds on compact subsets).
Convergence needs `k -> 0` and `h / k -> 0`: for a fixed `m` the error stops falling once the interpolation error, of order `(h / k)^2`, dominates, so the paper's best `m` grows with the refinement (from 2 to 16 on T39, 4 to 64 on T40).

### Solution

`solution` returns the nodal values, the piecewise linear interpolant as `u` (outside `Omega_h`, in slivers of width `O(h^2)` along a curved boundary, the value at the nearest point of `Omega_h`'s boundary, the paper's constant extension along the normal, its Figure 1), and as `node_hessian` at each unknown the least-squares fit of a symmetric matrix to its second differences along every direction (`fit_hessian`).
The fit can be indefinite where every second difference is nonnegative (Known limits), so `evaluate`'s `nonconvex_fraction` and `eigmin` measure the fit, not the convexity of the solution.
`info` holds `h`, `residual_max`, `min_reach` (the smallest stencil size over `h`) and `interpolant` (`"linear"`).

## Howard's algorithm

`Howard` starts from `laplace u = 2 sqrt(f)`, which for this scheme is the equation of the control `B = Id / 2` (the paper's start `B_0 = Id / d`), and takes full Newton steps `v - J^{-1} r(v)`, stopping when a step changes no unknown by more than `tol` (the paper's `10^-6`).
`SemiLagrangian.jacobian` differentiates the maximising frame with its split `a` (Danskin's theorem): it is the linear part `L^B` of the maximising control, an M-matrix (Lemma 5.1).
Because the equation of each control is affine in `u`, a Newton step solves the equation of the control that maximises at the current iterate: this is Howard's policy iteration (Algorithm 1), which converges from any start, monotonically decreasing and superlinearly (Theorem 5.2).
`PolicyIteration`, the iteration of the mixed scheme (`docs/methods/chen_wan_lin2018.md`), is the same algorithm for a scheme that returns each policy's system (`policy_system`) and stops on the residual; `Howard` works from the scheme's `residual` and `jacobian` and stops on the step, the paper's rule.
In the reproduction it takes 4 to 7 steps on T39 and 2 to 9 on T40, against 4 to 16 in the paper's Figure 8; on T40 turned off the grid, more where the stencil is narrow, up to 27 at `m = 2` (`benchmarks/feng_jensen2017.md`).

The linear systems are wide-stencil M-matrices, which a sparse LU factorisation fills in (35 s at `n = 121`, `m = 32`, against 0.01 s for the Krylov solve); `Howard` solves them by BiCGSTAB preconditioned with the diagonal, then restarted GMRES, then the LU factorisation where a Krylov method stops short of a relative residual of `1e-8`.
The Krylov methods need fewer steps the wider the stencil, since the condition number grows like `(diam / k)^2`.

`Howard` takes any scheme with `residual`, `jacobian` and `solve_laplace`, so `Howard(scheme=Standard9(n))` is undamped Newton, whose Jacobian is not an M-matrix, so its steps rest on the LU fallback; and `Newton(scheme=SemiLagrangian(n))` solves the semi-Lagrangian scheme too, with a line search, reaching the same (unique) root, but `Newton` factorises by LU, which is much slower on wide stencils (Known limits).

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `SemiLagrangian` | `n` | 64 | cells per side of the grid over the bounding box, at least 2 |
| | `stencil_factor` | 4.0 | `m` in `k = m h`, positive |
| | `directions` | 64 | stencil directions over `[0, pi)`, even and at least 4; the paper's 64 |
| `Howard` | `max_iter` | 50 | most steps |
| | `tol` | `1e-6` | largest change of an unknown in the last step; the paper's |

`FDSemiLagrangian(n, stencil_factor, directions, max_iter, tol)` is `Howard(scheme=SemiLagrangian(n, stencil_factor, directions), max_iter, tol)`.
The default `m = 4` is the paper's best `m` on T39 around 5,000 nodes; on a nonsmooth problem a wider stencil does better at fine grids.

## Known limits

- Structured nodes by default, not the paper's unstructured grid; the triangulation is a Delaunay triangulation of the grid nodes and boundary points, and the 0.3 finite-element infrastructure will bring meshes.
  The alignment matters: T40's kink `x1 = 0` runs along the grid columns and along the stencil direction `(0, 1)`, so the interpolant of `|x1|` solves the scheme and the error at the nodes is at the rounding level for every `n` (`benchmarks/feng_jensen2017.md`, Finding 1).
- Convex 2D domains only (`Box`, `Disk`, `Ellipse`, `CutQuarterDisk`, `DiskUnionSquare`) with a Dirichlet boundary.
- The accuracy depends on `stencil_factor`, and the best value depends on the grid and the problem; the error at a fixed `m` levels off as `h` falls.
- Memory and setup grow with `directions`: the scheme stores 7 entries per unknown and direction, and locates `2 directions` stencil ends per unknown in the triangulation; a solve at `n = 1280` with 64 directions holds about 10 GB.
- Time grows with `n` at a fixed `stencil_factor`, since the Krylov methods need more steps the smaller `k / diam`: on a100server, one thread, a solve of T40 turned off the grid at `m = 2` took 82 s at `n = 320` and 1767 s at `n = 640` (25 and 27 Howard steps), and more than 13 hours at `n = 1280` (`benchmarks/feng_jensen2017.md`, Rerun): each Krylov method has a budget of `10 N` steps before the LU fallback, which keeps that fallback out of reach; #72 proposes a cap.
- `node_hessian` is a least-squares fit, and a fit to nonnegative second differences can be indefinite: one unit difference and the others 0 fit the eigenvalues `-1/D` and `3/D` for `D` directions.
  It is worst where the differences are nearly all 0, in flat regions and next to kinks: on T40 the solution is exact to rounding, yet the fit gives a quarter of the nodes an eigenvalue below `-1e-6` at `n = 80`, `m = 16`, down to -2.3; on T20 the comparison table's 8.8% of non-convex nodes comes mostly from the fit: at the 95% of the nodes where every second difference is at least `-1e-8`, 4% still have a fitted eigenvalue below `-1e-6`, down to -2.5e-2, while the smallest second difference, a trace of Howard's tolerance, is -8.1e-4 ([`benchmarks/feng_jensen2017.md`](../../benchmarks/feng_jensen2017.md), Finding 3).
  So `evaluate`'s `nonconvex_fraction` and `eigmin` measure the fit for this scheme; #73 proposes the Hessian of the maximising frame, whose eigenvalues are the frame's second differences.
  Where `k < h` next to the boundary the second differences behind the fit are not consistent either.
- `Newton(scheme=SemiLagrangian(n))` solves the linear systems by LU, which fills in on wide stencils: on T39 it took 64 s against 1.3 s for `FDSemiLagrangian` at `n = 128`, `m = 8`. Use `Howard` or the preset; #72 proposes to let the scheme choose its linear solve.
- T39 and T40 lie on `DiskUnionSquare`, which only this scheme supports for now; the other presets refuse them (#74).
