# Monotone mixed scheme (Chen-Wan-Lin 2018)

The monotone mixed scheme of Chen, Wan and Lin (2018) solves the Monge-Ampere equation in its HJB form, a maximum over controls of linear operators, and discretises each control with the standard 7-point stencil where that stencil is monotone and with the semi-Lagrangian wide stencil elsewhere.
Every control's operator then has an M-matrix, and the paper proves convergence to the viscosity solution; where the 7-point stencil serves every node the scheme is second order.
maesolvers ships it as the scheme `MixedHJB`, the iteration `PolicyIteration` (Howard's algorithm, the paper's), and the preset `FDMixed` for the pair (ADR 0004).
Reproduction: [`benchmarks/chen_wan_lin2018.md`](../../benchmarks/chen_wan_lin2018.md).

```python
from maesolvers import T2, FDMixed, MixedHJB, PolicyIteration
from maesolvers.metrics import evaluate

result = FDMixed(n=64).solve(T2)  # the preset
same = PolicyIteration(scheme=MixedHJB(n=64)).solve(T2)
print(evaluate(result)["max"])  # 8.211e-5, Chen-Wan-Lin's Table 2 at N = 64
```

## The HJB form

For a convex `u`, `det D^2 u = f` is equivalent to (the paper's Theorem 1, after Krylov 1972 and Lions 1984)

```text
max over A of ( -tr(A D^2 u) + 2 sqrt(det(A) f) ) = 0,
A symmetric positive semidefinite with tr A = 1,
```

since `tr(A H) >= 2 sqrt(det A det H)` for positive semidefinite `A` and `H`, with equality at `A` proportional to `H^-1`.
The control `A` is parametrised by `(a, theta)` in `[0, 1] x [-pi/4, pi/4)`: eigenvalues `a` and `1 - a` along the axes `e_z = (cos theta, -sin theta)` and `e_w = (sin theta, cos theta)`, so (eq. (6) to (8))

```text
a11 = (1 - (1 - 2a) cos 2 theta) / 2,   a22 = (1 + (1 - 2a) cos 2 theta) / 2,   a12 = (1 - 2a) sin 2 theta / 2,
L_(a, theta) u = -a11 u_xx - 2 a12 u_xy - a22 u_yy + 2 sqrt(a (1 - a) f)
              = -a u_zz - (1 - a) u_ww + 2 sqrt(a (1 - a) f).
```

For a fixed control `L` is linear in `u`, and the maximum over the controls is 0 exactly at a convex solution: convexity is built in, with no constraint and no branch to select.
`control_coefficients(a, theta)` in `maesolvers.classical.mixed` computes `a11, a22, a12`.

## The discretisation

The domain is a box with equal sides and `n` cells per side (`h = 1/n` on the unit square, ADR 0001); the unknowns are the values at the `(n - 1)^2` interior nodes.
`u_xx` and `u_yy` are the centred 3-point differences.
The stencil of `u_xy` depends on the control (the paper's Section 4):

- where `a11 >= |a12|`, `a22 >= |a12|` and `a12 >= 0` (condition (12)), the 7-point difference with the diagonal `(1, 1)`, eq. (13); where the same holds with `a12 <= 0` (condition (14)), the one with `(1, -1)`, eq. (15).
  Either way `L` has nonpositive off-diagonal weights, and the stencil is second order.
  `seven_point_case(a, theta)` says which applies; the controls that meet one of them are the band `C^- <= a <= C^+` of `seven_point_band(theta)`, eq. (37): all of `[0, 1]` at `theta = 0` and `-/+ pi/4`, where the turned axes are grid axes or diagonals, and `[0.146, 0.854]` at `theta = -/+ pi/8`.
- every other control takes the semi-Lagrangian wide stencil, eq. (16) to (19): `u_zz` and `u_ww` as centred second differences along `e_z` and `e_w` with arms of length `sqrt(h)`, the values at the arms' ends interpolated bilinearly from the four nodes around them.
  The arm length makes the interpolation error `O(h^2) / h = O(h)`, so the stencil is first order, and it has at most 17 points.
  An arm that leaves the box is shortened to it, and the Dirichlet data at the boundary point replace the interpolated value, with the three-point difference of unequal arms.

At every node the discrete equation is the maximum over the controls of `L` with the control's own stencil, and policy iteration solves it (below).

### The optimal control

The maximum over the 7-point controls is closed-form (Section 5.2): the stationary point of each region, eq. (30), kept only if it lies in its region; the line `theta = 0`, eq. (31); and the boundaries with the wide-stencil region, eq. (33) and (34).
`seven_point_controls(dxx, dyy, dxy1, dxy2, f)` returns the best of these seven candidates, the control, the stencil, and where a stationary point lies in its region.
Over the wide-stencil controls, the best `a` for each `theta` is closed-form, eq. (36) to (38) (`wide_stencil_weight`), and `theta` is searched over `M = angles` values `-pi/4 + k pi / (2M)`, `k = 0, ..., M - 1`; by default `M = n`, the paper's `M = N`, so the scheme stays consistent as `h` goes to 0.

Where the stationary point of a 7-point objective lies in its region, it maximises that objective over all controls, and the paper's cost estimate (`O(1)` per node where the 7-point stencil applies) says its algorithm takes it without the wide-stencil search.
The published tables match only this rule (`benchmarks/chen_wan_lin2018.md`, Finding 2), so it is the default, `wide_search="outside"`: the wide-stencil controls compete only at the nodes where neither stationary point lies in its region.
`wide_search="everywhere"` lets them compete at every node, as the paper's Section 5.2 states; `wide_search="only"` takes the wide stencil for every control, with `a` unrestricted, which is the pure semi-Lagrangian scheme of Feng and Jensen (2017) that the paper compares with.

## Policy iteration

`PolicyIteration` is Howard's algorithm (the paper's Section 5.1): at the current unknowns it finds the optimal control at every node, assembles that policy's linear system `A v = F` (the scheme's `policy_system(grid, v)`), stops if `max |A v - F| <= tol`, and otherwise solves the system for the next unknowns.
`A` is an M-matrix for every policy, so the iteration converges from any start (Bokanowski, Maroso and Zidani 2009); it is Newton's method with full steps and the policy's matrix as the Jacobian.
It starts from `laplace u = 2 sqrt(f)`, the solution for `a = 1/2` (eq. (50)), and the paper's `tol = 1e-6` is the default.
`Result.info["history"]` holds the residual at each iterate before its update, and `iterations` counts the policies evaluated, the paper's "number of policy iterations".
`Newton(scheme=MixedHJB(...))` also solves the scheme, as a damped policy iteration, since `MixedHJB.jacobian` is the optimal policy's matrix.

A scheme of your own in HJB form plugs into `PolicyIteration` by providing `policy_system(grid, v) -> (A, F)` and `solve_laplace(grid, rhs)` besides the four methods of `Scheme`.

## Source paper

- Y. Chen, J. W. L. Wan, J. Lin, "Monotone mixed finite difference scheme for Monge-Ampere equation", J. Sci. Comput. 76 (2018) 1839-1867, doi 10.1007/s10915-018-0685-y: the HJB form (Section 3), the mixed discretisation (Section 4), policy iteration and the optimal controls (Section 5), the convergence proof (Section 6), and Tables 1 to 5 on T1 over `[-1, 1]^2`, T2, T20c, the cone T21 and `f = 1`, `g = 0` over `(-0.5, 0.5)^2`.
  The roadmap and the research survey call it "Chen-Wan-Wang 2018"; the title page names Jessey Lin.
  It is a mixed 7-point and semi-Lagrangian scheme, not a semi-Lagrangian scheme alone.
- X. Feng, M. Jensen, "Convergent semi-Lagrangian methods for the Monge-Ampere equation on unstructured grids", SIAM J. Numer. Anal. 55 (2017) 691-712: the pure semi-Lagrangian scheme, `wide_search="only"`.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `MixedHJB` | `n` | 64 | cells per side, at least 2; the paper's `N` |
| `MixedHJB` | `angles` | None (`n`) | the number `M` of angles of the wide-stencil search, at least 1 |
| `MixedHJB` | `wide_search` | `"outside"` | where the wide-stencil controls compete: `"outside"` (the paper's tables), `"everywhere"` (its Section 5.2), `"only"` (the pure semi-Lagrangian scheme) |
| `MixedHJB` | `spline_degree` | 3 | degree of the spline off the nodes, 1 to 5; degree `n` if `n` is smaller |
| `PolicyIteration` | `max_iter` | 100 | most policy evaluations |
| `PolicyIteration` | `tol` | 1e-6 | largest residual `max \|A v - F\|` of the HJB equation, absolute |
| `PolicyIteration` | `krylov_tol` | None | None: each policy's system by sparse LU; a value: by BiCGSTAB to that relative residual, preconditioned by an incomplete LU (drop tolerance `1e-4`), as the paper's GMRES is |

The preset takes `n`, `angles`, `spline_degree`, `max_iter`, `tol` and `krylov_tol` flat: `FDMixed(n=128)` is `PolicyIteration(scheme=MixedHJB(n=128))`.
Results carry `info["residual_max"]`, `info["wide_fraction"]` (the share of unknowns whose optimal control takes the wide stencil at the final iterate) and `info["angles"]`.

## Known limits

- A box with equal sides and Dirichlet data only, in 2D; `f` must be finite and nonnegative at the interior nodes.
- Where the wide stencil enters, the scheme is first order at most, and its error does not fall regularly with `h` (T20c, T21; the paper's Section 7 says the same).
- With the default rule the scheme switches between the 7-point and the wide stencil discontinuously in `u` at a node where a stationary point leaves its region; the paper's convergence proof covers the maximum over all controls (`wide_search="everywhere"`), and on T2 that maximum takes the wide stencil near the corner `(1, 1)` and loses the second order (Finding 2).
- With the default rule the scheme can have more than one root: on T2 from `N = 256` on, policy iteration from the paper's start ends on a root with the wide stencil next to the corner, ten to sixty times less accurate than the 7-point root it reaches from the exact solution (`benchmarks/chen_wan_lin2018.md`, Finding 5).
- A Dirac mass needs a discretisation of its own: with `f = pi / h^2` at the tip of the cone the error does not decrease with `h` (`benchmarks/chen_wan_lin2018.md`, Finding 3).
- The search over the angles costs `O(M)` per node where the wide stencil competes, some 100 seconds per policy at `n = 512` with `wide_search="only"`.
- The sparse LU of a policy with many wide-stencil rows fills in: at `n = 512` it takes tens of GB and does not finish in hours, so give `krylov_tol` there (`benchmarks/chen_wan_lin2018.md` uses `1e-12` from `n = 256` on).
- `node_hessian` holds the centred 9-point differences, which the scheme does not use.
