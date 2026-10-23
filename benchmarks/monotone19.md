# Monotone 19-point scheme in 3D: published versus ours

Problems, on `(0, 1)^3` with Dirichlet data from the exact solution, `x0 = (0.5, 0.5, 0.5)`: T9 (the `C^2` example (14), `u = exp(|x|^2 / 2)`), the `C^1` example (15) (`u = ((|x - x0| - 0.2)^+)^2 / 2`, `f = 0` on the ball of radius 0.2; T20 in 3D), and T22 (the example with blow-up (16), `u = -sqrt(3 - |x|^2)`, `f` infinite at the corner `(1, 1, 1)`).
The `C^1` example has no T id in the survey or the shared test-set specification, so it is not in the catalogue: `run_monotone19.py` defines it as `C1_3D`, and `tests/test_reproduction.py` checks that its `f` is `det D^2 u` (issue #27 asks JC for an id).
Published numbers: Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 4 (SINUM below), max error, Newton iterations and CPU time, checked against the page image and transcribed in `sources.md`.
The paper computes "using a 19 point stencil on an `N^3` grid" with Newton's method (Section 7) and states neither the stopping rule nor whether `N` counts nodes or cells.

Ours: `FDMonotone(n=n, stencil=19)`, the preset `Newton(scheme=Monotone17(n=n, stencil=19), max_iter=200)` (ADR 0004), residual tolerance `1e-10` relative to `max(1, max f)`, float64, scipy sparse LU, with `n = N` cells per side (`h = 1/N`, ADR 0001) and with `n = N - 1` (`h = 1/(N - 1)`, `N` nodes per side).
The scheme differs from the paper's as in 2D (`monotone17.md`, "Ours"): each triple's product continues below zero with the slope `kappa = max(f, mean f, 1)^(2/3)`, which changes no discretely convex grid function's value; `monotone19_checks.csv` runs the bare scheme to test it.
Every run converged.
Error: max over all grid nodes of `|u_h - u|`.
Raw data: `monotone19.csv` and `monotone19_iterates.csv`, written on a100server by `run_monotone19.py` at commit 164c1b8 (`just remote-bg monotone19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/run_monotone19.py`, tmux session `maesolvers-monotone19`), and `monotone19_checks.csv` by `run_monotone19.py --checks` at 4e854b6 (session `maesolvers-monotone19-checks-r1`).
Every solve is one run in the MLflow experiment `maesolvers-monotone19-<problem>` (T9, C1-3D, T22), tagged with its commit and a clean checkout.

"As printed" rounds our value to the digits the paper prints; a bold value differs from the published one there.
The levels follow ADR 0002; the convergence order of a column is the least-squares slope of `-log error` against `log N` over the printed `N`; a ratio is ours divided by published.

## Summary

The level of each problem is the one with `h = 1/N`, the grid convention of ADR 0001, which the same paper's 2D Table 3 follows (`monotone17.md`):

| Problem | Level (`h = 1/N`) |
|---|---|
| T9 | order of magnitude; the printed digits at `N = 21` and 31 |
| `C^1` example (15) | order of magnitude (Finding 3) |
| T22 | mismatch: orders differ by 0.24, from `N = 7` alone (Finding 2) |

The paper does not say whether its `N` counts cells or nodes, and `h = 1/(N - 1)` fits some cells better (Finding 1); the coarsest grid `N = 7` agrees with no convention (Finding 2).
`tests/test_reproduction.py` pins the one cell that agrees to the printed digits with `h = 1/N` and solves in seconds, T9 at `N = 21`; `N = 31` agrees too but takes a minute.

The 3D hybrid of Froese-Oberman, J. Comput. Phys. 230 (2011), Table 4 (`sources.md`) is not reproduced: maesolvers has no 3D standard scheme yet (`Standard9` is 2D; a 3D centred scheme comes with the multigrid of #30), so `Hybrid` stays 2D.
Its printed errors agree with SINUM's at `N = 7` on all three problems and at `N = 11` on T9 and the `C^1` example, as expected where the band of one grid step next to the boundary covers nearly every node and the hybrid scheme is the monotone one; they come from separate runs, since the CPU times printed next to them differ (at `N = 7`, T9 0.1 s against 0.04 s, T22 0.02 s against 0.03 s).

## T9, the `C^2` example (14)

| N | Published | `h = 1/N` | as printed | `h = 1/(N - 1)` | as printed | Newton steps (published, ours with `h = 1/N`) |
|---|---|---|---|---|---|---|
| 7 | 0.0151 | 0.015828 | **0.0158** | 0.016942 | **0.0169** | 2, 4 |
| 11 | 0.0140 | 0.013685 | **0.0137** | 0.013999 | 0.0140 | 3, 4 |
| 15 | 0.0132 | 0.013005 | **0.0130** | 0.013172 | 0.0132 | 5, 4 |
| 21 | 0.0127 | 0.012683 | 0.0127 | 0.012672 | 0.0127 | 6, 4 |
| 31 | 0.0125 | 0.012460 | 0.0125 | 0.012449 | **0.0124** | 5, 6 |

Ratio 0.98 to 1.05 (`h = 1/N`) and 1.00 to 1.12 (`h = 1/(N - 1)`); orders: published 0.13, ours 0.16 and 0.20.
Level: order of magnitude under either convention.
The error stops decreasing near `1.25e-2`: the 19-point stencil's angular resolution, `acos(sqrt(2/3))`, about 35 degrees (`angular_resolution(19)`), limits it, not `h`, as the 9-point stencil does in 2D (`monotone17.md`, T1).
The steps are ours from our start (`laplace u = 3 f^(1/3)`), recorded, not judged.

## The `C^1` example (15)

| N | Published | `h = 1/N` | `h = 1/(N - 1)` | Newton steps (published, ours with `h = 1/N`) |
|---|---|---|---|---|
| 7 | 0.0034 | 0.0050 | 0.0059 | 1, 4 |
| 11 | 0.0022 | 0.0037 | 0.0041 | 1, 7 |
| 15 | 0.0019 | 0.0032 | 0.0031 | 1, 5 |
| 21 | 0.0020 | 0.0028 | 0.0027 | 2, 16 |
| 31 | 0.0019 | 0.0025 | 0.0026 | 2, 87 |

Ratio 1.34 to 1.70 (`h = 1/N`) and 1.34 to 1.85 (`h = 1/(N - 1)`); orders: published 0.36, ours 0.45 and 0.58.
Level: order of magnitude with `h = 1/N`; mismatch with `h = 1/(N - 1)`, whose order differs by 0.22.
Our error is above the published one at every `N` (Finding 3).
Newton converges slowly here, 87 steps at `N = 31` (122 with `h = 1/(N - 1)`), about 12 minutes on one core, where T9 and T22 take 6 to 10.
In the ball where `f = 0` two factors of a triple can vanish together, and the Jacobian's coefficient is then the square of the floor.
`monotone19_checks.csv` (rows `floor` and `floor once`, `h = 1/N`) varies the floor: its size (the paper's `1e-8` against our `1e-12`) and its structure (`floor once`: the coefficient `max(prod_{j != k} a_j^+, floor)` in place of the product of floored factors, the same in 2D):

| N | Steps, floor `1e-12` (ours) | floor `1e-8` | floor once, `1e-12` |
|---|---|---|---|
| 15 | 5 | 13 | 11 |
| 21 | 16 | 14 | 7 |
| 31 | 87 | 29 | 52 |

Every variant reaches the same root: the node values differ from ours by at most `2.1e-7` (column `max_diff_default`), and the max errors agree to `3e-9`.
So the floor, both its size and its structure, changes how fast Newton converges here, but no variant is fastest at every `N`; why Newton is slow on this problem is not known (issue #65).

## T22, the example with blow-up (16)

In units of `1e-3`:

| N | Published | `h = 1/N` | as printed | `h = 1/(N - 1)` | as printed | Newton steps (published, ours with `h = 1/N`) |
|---|---|---|---|---|---|---|
| 7 | 9.6 | 6.598 | **6.6** | 7.110 | **7.1** | 1, 4 |
| 11 | 5.3 | 5.010 | **5.0** | 5.349 | 5.3 | 3, 5 |
| 15 | 4.7 | 4.463 | **4.5** | 4.561 | **4.6** | 3, 5 |
| 21 | 4.3 | 4.184 | **4.2** | 4.218 | **4.2** | 6, 7 |
| 31 | 3.9 | 4.028 | **4.0** | 4.026 | **4.0** | 8, 9 |

Ratio 0.69 to 1.03 (`h = 1/N`) and 0.74 to 1.03 (`h = 1/(N - 1)`); orders: published 0.57, ours 0.33 and 0.38.
Level: mismatch with `h = 1/N`, by the order rule alone (0.24 apart, the threshold is 0.2); order of magnitude with `h = 1/(N - 1)` (0.19 apart).
From `N = 11` on the ratio is 0.95 to 1.03 under either convention; the gap in the orders comes from `N = 7` (Finding 2).
The centred Hessian of our node values has a negative eigenvalue at a few nodes next to the corner from `N = 15` on (`nonconvex_fraction` 1e-4 to 5e-4), where it sees the blow-up, as on T2 in 2D.

## 2D unchanged

The 3D scheme rewrites the kernels for frames of `d` directions; every 2D result stays bit for bit.
`check_2d_bit_for_bit.py` compares 108 solves (every 2D finite-difference preset on T1, T2 and T20, the standard scheme also on T8 and T6, at `n = 8`, 16 and 25) and the difference operators between the reference 79a83b0 and this branch at 4e854b6, on a100server: `912 arrays, 0 differ` (log `monotone19-bits-r1.log` in the data dir).

## Finding 1: the printed `N` does not fix the grid convention

The paper's "`N^3` grid" in 3D could mean `N` cells per side (`h = 1/N`, ADR 0001), `N` nodes per side (`h = 1/(N - 1)`), or `N` interior nodes per side (`h = 1/(N + 1)`); in 2D the same paper's tables follow `h = 1/N` (ADR 0001, `monotone17.md`).
On T9 the printed digits are ours with `h = 1/N` at `N = 21` and 31, and with `h = 1/(N - 1)` at `N = 11`, 15 and 21; at `N = 31` the `h = 1/(N - 1)` value `0.012449` misses the printed `0.0125` by `1e-6`.
On T22 only `N = 11` with `h = 1/(N - 1)` agrees.
With `h = 1/(N - 1)` the levels would be: T9 order of magnitude, the `C^1` example mismatch (orders 0.36 published, 0.58 ours), T22 order of magnitude.
`h = 1/(N + 1)` fits worse (`monotone19_checks.csv`, rows `convention`): it gives the printed digits only on T9 at `N = 31`, where the error has stopped decreasing; T9 is order of magnitude (ratio 0.95 to 1.00), the `C^1` example order of magnitude (ratio 1.37 to 1.60, orders 0.36 and 0.44), and T22 a mismatch (ratio 0.64 to 1.03, orders 0.57 and 0.27).
No convention fits every cell; the summary gives the level with ADR 0001's `h = 1/N`, and only the `h = 1/N` agreement at `N = 21` is pinned.

## Finding 2: at `N = 7` the published errors match neither convention

At `N = 7`, the coarsest grid (5 or 6 interior nodes per side), the published errors are 5 to 11 percent below ours on T9, 35 to 45 percent above on T22, and 31 to 43 percent below on the `C^1` example, under either convention.
The paper reports 1 or 2 Newton steps there, so its values may be early iterates, as on the coarse 2D grids (`monotone17.md`, Finding 2); the JCP 2011 hybrid table prints the same three errors at `N = 7` from separate runs (its CPU times differ), as expected where the hybrid scheme is the monotone one.
`monotone19_iterates.csv` runs Newton from the paper's start, `laplace u = (6 f)^(1/3)` (SINUM eq. (8)), solved on the grid itself, for 1 to 8 steps: its iterates are far from the published values (on T22 at `N = 7` the first step has error `1.4e-2` under either convention, the second `0.70e-2` or `0.91e-2`), and it needs more steps than the paper reports at every `N`.
The paper solves its start on a coarse grid, takes its convex envelope where needed and interpolates it (Section 5.2); it states no grid, no envelope solver and no stopping rule, so its iterates cannot be rebuilt, and the early-stop explanation stays unproven.
T22's order mismatch with `h = 1/N` rests on this one grid: without `N = 7` the orders are 0.29 published, 0.21 ours with `h = 1/N` and 0.27 with `h = 1/(N - 1)`.

## Finding 3: on the `C^1` example the published errors stay below the convex discrete solution's

Our error is 1.34 to 1.85 times the published one at every `N` and under either convention, the reverse of T20 in 2D, where the published errors were above ours (`monotone17.md`, Finding 5).
Our continuation below zero makes the root convex along every stencil direction, so a non-convex root of the paper's bare scheme is a candidate, as it was in 2D.
`monotone19_checks.csv` (rows `bare`) runs Newton on the bare scheme `min prod (D u)^+ = f` with the paper's floored Jacobian, with `h = 1/N`:

| N | Published | Ours | Bare, our start | Bare, the paper's start |
|---|---|---|---|---|
| 7 | 3.4e-3 | 4.96e-3 | 4.96e-3 | 4.96e-3 |
| 11 | 2.2e-3 | 3.70e-3 | 4.73e-3 | 5.31e-3 |
| 15 | 1.9e-3 | 3.23e-3 | 3.82e-3 | 4.63e-3 |
| 21 | 2.0e-3 | 2.77e-3 | 3.46e-3 | line search stalls |

From `N = 11` on the bare scheme reaches non-convex roots (smallest second difference `-0.07` to `-0.29`), and their errors are larger than our convex root's, so they move away from the published values, not towards them.
At `N = 7` the bare roots are ours.
Neither a non-convex root nor the Jacobian floor (`C^1` section; it changes the steps, not the root) explains the gap; the paper's 1 or 2 Newton steps from its own start (Finding 2) remain a candidate we cannot test.
Our scheme has no root with a negative second difference in a triple (`monotone17.md`, "Below zero" holds in 3D too), so the published values, if they are converged, come from another discrete solution than this scheme's convex root.
