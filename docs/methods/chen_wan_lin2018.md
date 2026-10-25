# Monotone mixed scheme (Chen-Wan-Lin 2018)

The monotone mixed scheme of Chen, Wan and Lin (2018) solves the Monge-Ampere equation in its HJB form, a maximum over controls of linear operators, and discretises each control with the standard $7$-point stencil where that stencil is monotone and with the semi-Lagrangian wide stencil elsewhere.
Every control's operator then has an M-matrix; the maximum over all controls is monotone, and the paper proves its convergence to the viscosity solution; where the $7$-point stencil serves every node the scheme is second order.
The paper's tables follow a cheaper rule, the default here, that lets the wide stencil compete only at some nodes: that scheme is not monotone everywhere, can have a second root, and can make policy iteration cycle, on the catalogue's T4 already at the preset's default $n = 64$ (below, and [`benchmarks/chen_wan_lin2018.md`](../../benchmarks/chen_wan_lin2018.md), Findings 2, 5 to 7); `wide_search="everywhere"` is the monotone maximum.
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

For a convex $u$, $\det D^2 u = f$ is equivalent to (the paper's Theorem 1, after Krylov 1972 and Lions 1984)

$$
\begin{gathered}
\max_{A} \bigl( -\operatorname{tr}(A D^2 u) + 2 \sqrt{\det(A) f} \bigr) = 0, \cr
A \text{ symmetric positive semidefinite with } \operatorname{tr} A = 1,
\end{gathered}
$$

since $\operatorname{tr}(A H) \ge 2 \sqrt{\det A \det H}$ for positive semidefinite $A$ and $H$, with equality at $A$ proportional to $H^{-1}$.
The control $A$ is parametrised by $(a, \theta)$ in $[0, 1] \times [-\pi/4, \pi/4)$: eigenvalues $a$ and $1 - a$ along the axes $e_z = (\cos\theta, -\sin\theta)$ and $e_w = (\sin\theta, \cos\theta)$, so (eq. (6) to (8))

$$
\begin{aligned}
a_{11} &= (1 - (1 - 2a) \cos 2\theta) / 2, \quad a_{22} = (1 + (1 - 2a) \cos 2\theta) / 2, \quad a_{12} = (1 - 2a) \sin 2\theta / 2, \cr
L_{(a, \theta)} u &= -a_{11} u_{xx} - 2 a_{12} u_{xy} - a_{22} u_{yy} + 2 \sqrt{a (1 - a) f} \cr
&= -a u_{zz} - (1 - a) u_{ww} + 2 \sqrt{a (1 - a) f}.
\end{aligned}
$$

For a fixed control $L$ is linear in $u$, and the maximum over the controls is $0$ exactly at a convex solution: convexity is built in, with no constraint and no branch to select.
`control_coefficients(a, theta)` in `maesolvers.classical.mixed` computes $a_{11}, a_{22}, a_{12}$.

## The discretisation

The domain is a box with equal sides and $n$ cells per side ($h = 1/n$ on the unit square, ADR 0001); the unknowns are the values at the $(n - 1)^2$ interior nodes.
$u_{xx}$ and $u_{yy}$ are the centred $3$-point differences.
The stencil of $u_{xy}$ depends on the control (the paper's Section 4):

- where $a_{11} \ge \lvert a_{12} \rvert$, $a_{22} \ge \lvert a_{12} \rvert$ and $a_{12} \ge 0$ (condition (12)), the $7$-point difference with the diagonal $(1, 1)$, eq. (13); where the same holds with $a_{12} \le 0$ (condition (14)), the one with $(1, -1)$, eq. (15).
  Either way $L$ has nonpositive off-diagonal weights, and the stencil is second order.
  `seven_point_case(a, theta)` says which applies; the controls that meet one of them are the band $C^- \le a \le C^+$ of `seven_point_band(theta)`, eq. (37): all of $[0, 1]$ at $\theta = 0$ and $\mp \pi/4$, where the turned axes are grid axes or diagonals, and $[0.146, 0.854]$ at $\theta = \mp \pi/8$.
- every other control takes the semi-Lagrangian wide stencil, eq. (16) to (19): $u_{zz}$ and $u_{ww}$ as centred second differences along $e_z$ and $e_w$ with arms of length $\sqrt{h}$, the values at the arms' ends interpolated bilinearly from the four nodes around them.
  The arm length makes the interpolation error $O(h^2) / h = O(h)$, so the stencil is first order, and it has at most $17$ points.
  An arm that leaves the box is shortened to it, and the Dirichlet data at the boundary point replace the interpolated value, with the three-point difference of unequal arms.

At every node the discrete equation is the maximum over the controls of $L$ with the control's own stencil, and policy iteration solves it (below).

### The optimal control

The maximum over the $7$-point controls is closed-form (Section 5.2): the stationary point of each region, eq. (30), kept only if it lies in its region; the line $\theta = 0$, eq. (31); and the boundaries with the wide-stencil region, eq. (33) and (34).
`seven_point_controls(dxx, dyy, dxy1, dxy2, f)` returns the best of these seven candidates, the control, the stencil, and where a stationary point lies in its region.
Over the wide-stencil controls, the best $a$ for each $\theta$ is closed-form, eq. (36) to (38) (`wide_stencil_weight`), and $\theta$ is searched over $M$ = `angles` values $-\pi/4 + k \pi / (2M)$, $k = 0, \dots, M - 1$; by default $M = n$, the paper's $M = N$, so the scheme stays consistent as $h$ goes to $0$.

Where the stationary point of a $7$-point objective lies in its region, it maximises that objective over all controls, and the paper's cost estimate ($O(1)$ per node where the $7$-point stencil applies) says its algorithm takes it without the wide-stencil search.
Tables 2 and 4 match only this rule (`benchmarks/chen_wan_lin2018.md`, Finding 2) and Tables 1 and 3 match either, so it is the default, `wide_search="outside"`: the wide-stencil controls compete only at the nodes where neither stationary point lies in its region.
On Table 5 the two rules give the same values where the default converges, but the default does not converge at $N = 256$ (Finding 6), so Table 5 is a mismatch.
Which controls compete then depends on $u$, so the discrete equation is no longer a maximum of monotone operators over one control set: it can have more than one root, and policy iteration on it can cycle (Finding 5).
`wide_search="everywhere"` lets them compete at every node, as the paper's Section 5.2 states; `wide_search="only"` takes the wide stencil for every control, with $a$ unrestricted, which is the pure semi-Lagrangian scheme of Feng and Jensen (2017) that the paper compares with.

## Policy iteration

`PolicyIteration` is Howard's algorithm (the paper's Section 5.1): at the current unknowns it finds the optimal control at every node, assembles that policy's linear system $A v = F$ (the scheme's `policy_system(grid, v)`), stops if $\max \lvert A v - F \rvert \le \mathrm{tol}$, and otherwise solves the system for the next unknowns.
$A$ is an M-matrix for every policy; with `wide_search="everywhere"` or `"only"` the policies range over one set at every iterate, so the iteration converges from any start (Bokanowski, Maroso and Zidani 2009).
With the default `"outside"` that set depends on the iterate and the guarantee does not hold: in the receipts policy iteration cycles on T2 at $N = 512$, on Table 5's problem at $N = 256$ and on T4 from $n = 48$ on, and stops at `max_iter` (Known limits).
It is Newton's method with full steps and the policy's matrix as the Jacobian.
It starts from $\Delta u = d f^{1/d}$, as `Newton` does, in 2D $\Delta u = 2 \sqrt{f}$, the solution for $a = 1/2$ (eq. (50)), and the paper's $\mathrm{tol} = 10^{-6}$ is the default.
`Result.info["history"]` holds the residual at each iterate before its update, and `iterations` counts the policies evaluated, the paper's "number of policy iterations".
`Newton(scheme=MixedHJB(...))` also solves the scheme, as a damped policy iteration, since `MixedHJB.jacobian` is the optimal policy's matrix.

A scheme of your own in HJB form plugs into `PolicyIteration` by providing `policy_system(grid, v) -> (A, F)` and `solve_laplace(grid, rhs)` besides the four methods of `Scheme`.

## Source paper

- Y. Chen, J. W. L. Wan, J. Lin, "Monotone mixed finite difference scheme for Monge-Ampere equation", J. Sci. Comput. 76 (2018) 1839-1867, doi 10.1007/s10915-018-0685-y: the HJB form (Section 3), the mixed discretisation (Section 4), policy iteration and the optimal controls (Section 5), the convergence proof (Section 6), and Tables 1 to 5 on T1 over $[-1, 1]^2$, T2, T20c, the cone T21 and $f = 1$, $g = 0$ over $(-0.5, 0.5)^2$.
  The roadmap and the research survey call it "Chen-Wan-Wang 2018"; the title page names Jessey Lin.
  It is a mixed $7$-point and semi-Lagrangian scheme, not a semi-Lagrangian scheme alone.
- X. Feng, M. Jensen, "Convergent semi-Lagrangian methods for the Monge-Ampere equation on unstructured grids", SIAM J. Numer. Anal. 55 (2017) 691-712: the pure semi-Lagrangian scheme, `wide_search="only"`.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `MixedHJB` | `n` | $64$ | cells per side, at least $2$; the paper's $N$ |
| `MixedHJB` | `angles` | None (`n`) | the number $M$ of angles of the wide-stencil search, at least $1$ |
| `MixedHJB` | `wide_search` | `"outside"` | where the wide-stencil controls compete: `"outside"` (the paper's tables), `"everywhere"` (its Section 5.2), `"only"` (the pure semi-Lagrangian scheme) |
| `MixedHJB` | `spline_degree` | $3$ | degree of the spline off the nodes, $1$ to $5$; degree $n$ if $n$ is smaller |
| `PolicyIteration` | `max_iter` | $100$ | most policy evaluations |
| `PolicyIteration` | `tol` | $10^{-6}$ | largest residual $\max \lvert A v - F \rvert$ of the HJB equation, absolute |
| `PolicyIteration` | `krylov_tol` | None | None: each policy's system by sparse LU; a value: by BiCGSTAB to that relative residual, preconditioned by an incomplete LU (drop tolerance $10^{-4}$), as the paper's GMRES is; where BiCGSTAB fails, a `RuntimeWarning` and the sparse LU |

The preset takes `n`, `angles`, `wide_search`, `spline_degree`, `max_iter`, `tol` and `krylov_tol` flat: `FDMixed(n=128)` is `PolicyIteration(scheme=MixedHJB(n=128))`, and `FDMixed(n=128, wide_search="everywhere")` the monotone maximum.
Results carry `info["residual_max"]`, `info["wide_fraction"]` (the share of unknowns whose optimal control takes the wide stencil at the final iterate) and `info["angles"]`.

## Known limits

- A box with equal sides and Dirichlet data only, in 2D; $f$ must be finite and nonnegative at the interior nodes.
- Where the wide stencil enters, the scheme is first order at most, and its error does not fall regularly with $h$ (T20c, T21; the paper's Section 7 says the same).
- With the default rule the scheme switches between the $7$-point and the wide stencil discontinuously in $u$ at a node where a stationary point leaves its region; the paper's convergence proof covers the maximum over all controls (`wide_search="everywhere"`), and on T2 that maximum takes the wide stencil near the corner $(1, 1)$ and loses the second order (Finding 2).
- With the default rule the scheme can have more than one root: on T2 from $N = 256$ on, policy iteration from the paper's start ends on a root with the wide stencil next to the corner, $10$ to $65$ times less accurate than the $7$-point root it reaches from the exact solution (`benchmarks/chen_wan_lin2018.md`, Finding 5).
- With the default rule policy iteration can cycle between policies and stop at `max_iter` unconverged, and not only on fine grids: in the receipts on T2 at $N = 512$ (residual near $0.09$), on Table 5's problem at $N = 256$ (residual near $0.07$, also when started from the root of `wide_search="everywhere"`, Finding 6), and on the catalogue's T4 at every $n$ from $48$ to $128$, the preset's default $n = 64$ included, with residuals of $1.5$ to $7.9$ (Finding 7); `Result.info["converged"]` is then False.
  `wide_search="everywhere"` converges on all of them: on Table 5's problem at every $N$ ($18$ policy iterations at $N = 256$), to the default's values where both converge, and on T4 in $6$ to $13$ policy iterations.
  `FDMixed(n=n, wide_search="everywhere")` is the choice for a problem outside the paper's tables.
- The wide stencil's arm is the paper's $\sqrt{h}$ in the box's units, which does not scale with the box: on a box of side $L$ it spans $\sqrt{n / L}$ cells, so the resolution of the wide stencil is set by $n / L$, and a box of side $L$ needs about $L$ times the cells of the unit square.
  On $u = x^T H x / 2 + x_1$ with $H = [[1, 1.5], [1.5, 4]]$, outside the $7$-point band, `FDMixed` gives a max error relative to $\max \lvert u \rvert$ of $5.8 \times 10^{-3}$ and $2.2 \times 10^{-3}$ at $n = 16$ and $32$ on a box of side $1$, and $9.3 \times 10^{-3}$ at both on a box of side $10$, where the bilinear interpolation over arms of $1.3$ and $1.8$ cells overestimates the convex $u$ so much that the wide stencil wins at no node.
  `tests/test_mixed.py` (`test_the_wide_stencil_resolution_is_set_by_cells_per_side_of_the_box`) asserts the pattern: on side $1$ the error at $n = 32$ is less than half that at $16$, on side $10$ the two agree within $1$% and the wide stencil is used at no node.
  `MixedHJB.discretize` warns (`RuntimeWarning`) when the arm spans fewer than $2$ cells, $n < 4 L$.
- A Dirac mass needs a discretisation of its own: with $f = \pi / h^2$ at the tip of the cone the error does not decrease with $h$ (`benchmarks/chen_wan_lin2018.md`, Finding 3).
- The search over the angles costs $O(M)$ per node where the wide stencil competes, some $30$ seconds per policy at $n = 512$ with `wide_search="only"` on one thread; `MixedHJB.solution` runs it once more at the final iterate for `info["residual_max"]` and `info["wide_fraction"]`.
- The sparse LU of a policy with many wide-stencil rows fills in: at $n = 512$ it takes tens of GB and does not finish in hours, so give `krylov_tol` there; `benchmarks/chen_wan_lin2018.md` uses $10^{-12}$ from $n = 256$ on where the wide stencil is used at many nodes, and keeps the LU elsewhere, since that tolerance moves the fourth digit of a $7$-point error at $n = 512$.
- `node_hessian` holds the centred $9$-point differences, which the scheme does not use.
