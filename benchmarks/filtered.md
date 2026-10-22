# Filtered scheme: published versus ours

Problems (Froese-Oberman 2013, Section 4.2, on `[0, 1]^2` with Dirichlet data from the exact solution, `x0 = (0.5, 0.5)`): the `C^2` example `u = exp(|x - x0|^2 / 2)`, which is T1 centred at `x0` and not in the catalogue (called C2 below, `C2-centred` in the receipts; Finding 1); T20 (`C^1`, `f = 0` on a disk); T2 (corner blow-up, uncentred); and the cone T21 with `f = 4/h^2` at the nodes within `h/2` of `x0`, built per grid by `run_monotone17.cone`.
Published numbers: Froese-Oberman, SIAM J. Numer. Anal. 51 (2013), Table 4.1 (monotone and filtered, 9, 17 and 33 points) and Table 4.2 (17 points: filtered, its Newton iterations, and the standard scheme); `sources.md`, section 5, transcribes both, checked against the page images.
The hybrid columns of Table 4.2 belong to the hybrid scheme (`hybrid.md`) and are not compared here; `sources.md`, section 5, transcribes them.

Ours, with `n = N - 1` cells (Finding 1):

- filtered: `FDFiltered(n, stencil)`, the preset `Newton(scheme=Filtered(accurate=Standard9(n), monotone=Monotone17(n, stencil)), max_iter=200)`, filter width `sqrt(h) + dtheta / 10` with `dtheta` from Oberman 2008, eq. (3.3) (`docs/methods/filtered.md`);
- monotone: `FDMonotone(n, stencil)` (`monotone17.md`);
- standard: `FDPoisson(n)`, the standard 9-point scheme under the Poisson iteration, as the paper's "one of the methods of [BFO10]".

All with `boundary="narrow"` unless a row says otherwise, residual tolerance `1e-10` relative to `max(1, max f)` for Newton, float64, scipy sparse LU.
Error: max over all grid nodes of `|u_h - u|`.
Raw data: `filtered.csv`, written on a100server by `run_filtered.py` at commit 4130727 (`just remote-bg filtered2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/run_filtered.py --processes 16`), and `filtered_checks.csv` by `run_filtered.py --checks --processes 9` at commit 48d7eec (session `maesolvers-filtered3-checks`).
Every solve is one run in the MLflow experiment `maesolvers-filtered-<problem>` (`C2-centred`, T20, T2, T21, T1), tagged with that commit and a clean checkout; earlier generations were deleted.

A bold value differs from the published one at the printed digits; a star marks a solve that did not converge (Finding 4).
The levels follow ADR 0002; the order of a column is the least-squares slope of `log error` against `log N` over the printed `N`.

## Summary

| Problem | Table 4.1 monotone | Table 4.1 filtered | Table 4.2 filtered | Table 4.2 standard |
|---|---|---|---|---|
| C2 | 17 points exact; 9 and 33 order of magnitude, exact from `N = 63` | order of magnitude: exact but at `N = 63` (Finding 2) | exact | exact |
| T20 | order of magnitude (last digit) | mismatch (Finding 5) | mismatch (Finding 5) | exact |
| T2 | exact | 17 and 33 points order of magnitude, exact to `N = 127`, not converged; 9 points mismatch | order of magnitude, not converged | exact |
| T21 | mismatch (Finding 6) | mismatch | mismatch | exact |

`tests/test_reproduction.py` pins the filtered and standard values on C2 at `N = 31` and 63, the standard values on T2 and T20, and monotone values on C2 and T2, each to the printed digits, all with `n = N - 1`.

## C2, the centred smooth example

Table 4.1, in units of `1e-5`:

| N | 9 monotone | ours | 9 filtered | ours | 17 monotone | ours | 17 filtered | ours | 33 monotone | ours | 33 filtered | ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 9.45 | **9.44** | 4.54 | 4.54 | 9.12 | 9.12 | 4.54 | 4.54 | 9.38 | **9.12** | 4.54 | 4.54 |
| 63 | 4.91 | 4.91 | 1.04 | **1.06** | 3.42 | 3.42 | 1.04 | **1.06** | 3.40 | 3.40 | 1.04 | **1.06** |
| 127 | 3.79 | 3.79 | 0.26 | 0.26 | 1.67 | 1.67 | 0.26 | 0.26 | 1.39 | 1.39 | 0.26 | 0.26 |
| 255 | 3.51 | 3.51 | 0.06 | 0.06 | 1.17 | 1.17 | 0.06 | 0.06 | 0.66 | 0.66 | 0.06 | 0.06 |
| 361 | 3.48 | 3.48 | 0.03 | 0.03 | 1.08 | 1.08 | 0.03 | 0.03 | 0.51 | 0.51 | 0.03 | 0.03 |

Table 4.2, in units of `1e-5`:

| N | Filtered | ours | Newton steps, published | ours | Standard | ours |
|---|---|---|---|---|---|---|
| 31 | 4.54 | 4.54 | 2 | 3 | 4.54 | 4.54 |
| 63 | 1.06 | 1.06 | 2 | 3 | 1.06 | 1.06 |
| 127 | 0.26 | 0.26 | 2 | 3 | 0.26 | 0.26 |
| 255 | 0.06 | 0.06 | 2 | 3 | 0.06 | 0.06 |
| 361 | 0.03 | 0.03 | 2 | 4 | 0.03 | 0.03 |

The filter keeps the standard scheme at every node (`monotone_fraction` 0 at every `N` and stencil), so the filtered solution is the standard scheme's, `4.5406e-5` to `3.157e-7`, with order 2.02 (published 2.04); both boundary treatments give the same.
The monotone columns are ours to the printed digits but for `N = 31`, where the 9-point value is `9.441e-5` against `9.45e-5`, and the 33-point value `9.119e-5`, equal to the 17-point one, against `9.38e-5`; `boundary="interpolate"` gives `7.93e-5` there, so neither treatment explains it.
Ratio 0.97 to 1.00; orders: published 0.38, 0.86, 1.19; ours 0.38, 0.86, 1.18.

## T20, the `C^1` example

Table 4.1, in units of `1e-4`:

| N | 9 monotone | ours | 9 filtered | ours | 17 monotone | ours | 17 filtered | ours | 33 monotone | ours | 33 filtered | ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 21.54 | **21.64** | 3.73 | **3.71** | 8.66 | **8.70** | 3.99 | **3.92** | 6.39 | 6.39 | 3.67 | **3.92** |
| 63 | 21.33 | **21.32** | 1.51 | **1.73** | 6.82 | 6.82 | 1.40 | **1.85** | 3.18 | **3.17** | 1.46 | **1.85** |
| 127 | 21.55 | **21.54** | 0.92 | **2.01*** | 6.63 | 6.63 | 0.76 | **1.85*** | 2.49 | 2.49 | 0.78 | **1.93*** |
| 255 | 21.51 | **21.50** | 0.38 | **1.03*** | 6.58 | 6.58 | 0.46 | **0.56*** | 2.36 | 2.36 | 0.37 | **0.78*** |
| 361 | 21.53 | 21.53 | 0.23 | **6.76*** | 6.62 | 6.62 | 0.31 | **7.33*** | 2.37 | 2.37 | 0.28 | **6.13*** |

Table 4.2, in units of `1e-4`:

| N | Filtered | ours | Newton steps, published | ours | Standard | ours |
|---|---|---|---|---|---|---|
| 31 | 3.99 | **3.92** | 2 | 15 | 3.78 | 3.78 |
| 63 | 1.40 | **1.85** | 2 | 21 | 1.34 | 1.34 |
| 127 | 0.76 | **1.85*** | 3 | 28 | 0.59 | 0.59 |
| 255 | 0.46 | **0.56*** | 3 | 40 | -- | |
| 361 | 0.31 | **7.33*** | 6 | 88 | -- | |

Monotone: ratio 1.00 in every column, orders equal to two decimals; the 9-point column differs in the last printed digit at four of five `N`, the others at one.
These are the convex roots of `Monotone17`, and they match where FO 2011's monotone columns on the same problem did not (`monotone17.md`, Finding 5): the 2013 numbers support that finding's reading that the 2011 ones came from another root or another scheme.
Filtered: ratio 0.98 to 29; orders published 1.10, 0.99, 1.04, ours -0.03, 0.10, 0.08; mismatch (Finding 5).

## T2, the example with blow-up

Table 4.1, in units of `1e-3`:

| N | 9 monotone | ours | 9 filtered | ours | 17 monotone | ours | 17 filtered | ours | 33 monotone | ours | 33 filtered | ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 1.74 | 1.74 | 1.74 | 1.74* | 1.74 | 1.74 | 1.74 | 1.74* | 1.74 | 1.74 | 1.74 | 1.74* |
| 63 | 0.86 | 0.86 | 0.59 | 0.59* | 0.59 | 0.59 | 0.59 | 0.59* | 0.59 | 0.59 | 0.59 | 0.59* |
| 127 | 0.83 | 0.83 | 0.20 | 0.20* | 0.35 | 0.35 | 0.20 | 0.20* | 0.20 | 0.20 | 0.20 | 0.20* |
| 255 | 0.83 | 0.83 | 0.15 | **0.25*** | 0.33 | 0.33 | 0.13 | **0.15*** | 0.16 | 0.16 | 0.08 | **0.09*** |
| 361 | 0.83 | 0.83 | 0.17 | **0.28*** | 0.33 | 0.33 | 0.13 | **0.16*** | 0.15 | 0.15 | 0.08 | **0.09*** |

Table 4.2, in units of `1e-3`:

| N | Filtered | ours | Newton steps, published | ours | Standard | ours |
|---|---|---|---|---|---|---|
| 31 | 1.74 | 1.74* | 7 | 22 | 17.38 | 17.38 |
| 63 | 0.59 | 0.59* | 9 | 8 | 12.62 | 12.62 |
| 127 | 0.20 | 0.20* | 11 | 16 | 9.04 | 9.04 |
| 255 | 0.13 | **0.15*** | 8 | 20 | 6.43 | 6.43 |
| 361 | 0.13 | **0.16*** | 12 | 21 | 5.42 | 5.42 |

Monotone: exact, all three stencils.
Filtered: no solve converges (Finding 4), but the last iterates agree with the published values to the printed digits up to `N = 127`, with every stencil and both boundary treatments.
From `N = 255` on ours are larger: ratio 1.15 to 1.23 with 17 and 33 points, orders published 1.09 and 1.32, ours 1.00 and 1.26 (order of magnitude); with 9 points ratio up to 1.68 and orders 0.99 against 0.75 (mismatch).
The paper notes that with its `eps` the filtered scheme "is slightly less accurate on the blow-up example".
`Monotone17`'s slope below zero (`kappa = sqrt(max(f, mean f, 1))`, large near the corner, where the paper's operator has slope 1) is not the cause: with `kappa = 1` the errors are the same to four digits at `N = 63`, 127 and 255 (`filtered_checks.csv`, rows `kappa`).

## T21, the cone

Table 4.1, in units of `1e-3`:

| N | 9 monotone | ours | 9 filtered | ours | 17 monotone | ours | 17 filtered | ours | 33 monotone | ours | 33 filtered | ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 11.83 | **292.89** | 10.42 | **292.89*** | 3.56 | **690.60** | 4.16 | **690.52** | 1.61 | **1565.74** | 1.23 | **1565.28*** |
| 63 | 11.10 | **292.89** | 11.56 | **290.50*** | 3.49 | **690.90** | 2.82 | **690.87*** | 1.65 | **1565.71** | 1.90 | **1565.66*** |
| 127 | 11.80 | **292.89** | 10.97 | **292.89*** | 3.45 | **690.96** | 2.83 | **690.81*** | 1.64 | **1565.74** | 1.11 | **1547.70*** |
| 255 | 10.47 | **292.89** | 11.03 | **292.71*** | 3.46 | **690.98** | 3.06 | **690.96*** | 1.64 | **1565.74** | 1.15 | **1565.43*** |
| 361 | 10.40 | **292.89** | 10.37 | **292.87*** | 3.45 | **690.98** | 3.22 | **690.55*** | 1.64 | **1565.74** | 1.12 | **1285.95*** |

Table 4.2, standard column, in units of `1e-3`: published 5.19, 2.86, 1.54 at `N = 31`, 63, 127, ours 5.19, 2.86, 1.54.
Mismatch for the monotone and the filtered schemes (Finding 6); exact for the standard scheme.

## Finding 1: the `C^2` example is centred, and `N` counts nodes

Eq. (4.1) prints `u = exp(|x - x0|^2 / 2)` with `x0 = (0.5, 0.5)` (checked on the page image), although the text calls the four examples those of FO 2011, whose `C^2` example is T1, `exp(|x|^2 / 2)`.
And the paper's `N x N` grid has `N` nodes per side: with `n = N - 1` cells every value of the Table 4.2 standard column is ours to the printed digits, on all four problems (C2: 4.5406, 1.0641, 0.2577, 0.0634, 0.0316 in units of `1e-5`; T20: 3.785, 1.340, 0.587 in `1e-4`; T2: 17.384, 12.623, 9.038, 6.430, 5.417 in `1e-3`; T21: 5.192, 2.860, 1.542 in `1e-3`), and with `n = N` none is (C2 at `N = 31`: `4.246e-5`; T2: `17.15e-3`, the FO 2011 value; `filtered.csv`, rows `1/N`).
The monotone columns agree: on T2, `h = 1/(N - 1)` is the convention that the JCP 2011 monotone column follows at `N <= 63` (`monotone17.md`, Finding 3), and here it fits at every `N`.
This explains the disagreement noted in `papers.md` and the survey: Table 4.2's standard column (`4.54e-5` at `N = 31` on the "C^2 example", `17.38e-3` on T2) differs from FO 2011's (`7.14e-5`, `17.15e-3`) because it is another problem on another grid, not another discretisation; the pinned FO 2011 values stand.
The centred example has no T id; it is built in `run_filtered.py` and in `tests/test_reproduction.py`.

## Finding 2: Table 4.1 and Table 4.2 print different filtered values at `N = 63` on C2

Table 4.1 prints `1.04e-5` in all three filtered columns at `N = 63`; Table 4.2 prints `1.06e-5` for the same 17-point filtered solve, and `1.06e-5` for the standard scheme.
The filter keeps the standard scheme at every node there, so the filtered and standard values must agree; ours is `1.0641e-5` for both.
We read `1.04` as a misprint and compare the filtered column with Table 4.2; it is the only cell of Table 4.1 that disagrees with Table 4.2.

## Finding 3: `dtheta` and the filter width

The paper takes `eps = sqrt(h) + dtheta / 10` and gives `dtheta` no value; maesolvers takes Oberman's (2008, eq. (3.3)) angular resolution, half the largest gap between stencil directions (`pi/8`, `atan(1/2)/2`, `atan(1/3)/2`).
On C2 the choice does not matter (the filter keeps the standard scheme everywhere), and on T20 a wider filter does not move the error towards the published one: `dtheta` as the full gap, and twice and four times the width, give `1.733e-4` at `N = 63` against our `1.854e-4` and the published `1.40e-4` (`filtered_checks.csv`, rows `width`), see Finding 5.

## Finding 4: with the paper's Jacobian, Newton does not converge where the filter band holds the largest residual

In the band `1 < |F_A - F_M| / eps < 2` the filtered equation is `2 F_M - F_A + 2 eps sign(F_A - F_M)`, whose derivative is `2 J_M - J_A`; the paper takes `2 J_M`, dropping `-J_A`, because a negative weight on `J_A` "can lead to poorly conditioned or ill-posed linear systems" (p. 438).
Where a node in the band attains the largest residual the Newton direction need not decrease it, and `Newton`'s line search ends the solve, not converged, with the residual between `1e-2` and `1e-1` (relative to `max f`: `1e-4` on T2).
That happens on T2 at every `N` and stencil, on T20 from `N = 127` on, on the cone at nearly every `N`, and on T1 (below) at every `N`; on C2, where no node is in the band, and on T20 at `N <= 63`, Newton converges in 3 to 21 steps.
Without the damping it is no better, in exploratory runs not kept as receipts: undamped steps from our start converge linearly, with ratio about 0.87, on T2 at `n = 30`, and end with residuals of `4e2` and `4e3` after 300 steps at `n = 126` and 254; with the exact derivative `2 J_M - J_A` the steps diverge from the start.
The paper reports 2 to 12 Newton steps and states no start, no damping and no stopping rule, so its iterates may have stopped where ours do; on T2 our last iterates agree with its values to the printed digits up to `N = 127`.
Starting from the monotone or the standard solution does not help: Newton stalls there too (`filtered_checks.csv`, rows `start`).

On T1 itself (uncentred, `n = N`), which the paper does not run, `FDFiltered` never converges, and the last iterates have errors between `5.7e-5` and `1.4e-4` with 17 points and between `1.2e-5` and `1.5e-4` with 33 at `N = 31` to 361, mostly between the monotone scheme's (`4.4e-4` and `1.5e-4` at `N = 361`) and the standard one's (`7.1e-5` at `N = 31` to `5.3e-7` at 361); 1 to 14 percent of the nodes, near the corner `(1, 1)` where the Hessian is most anisotropic, leave the standard scheme (`filtered.csv`, rows `T1`).

## Finding 5: on T20 the filtered scheme inherits the standard scheme's non-convex root

Where the filter keeps the standard scheme the filtered equation is the standard one, and Newton on the standard 9-point scheme, from our start, reaches a non-convex discrete root on T20, as it does on T2 (`standard_fd.md`, Finding 2): `FDNewton` gives `1.733e-4` at `N = 63` and `1.992e-4` at 127, the convex root of the Poisson iteration `1.340e-4` and `0.587e-4` (`filtered_checks.csv`, rows `newton` and `width`).
With a filter two or four times wider every node keeps the standard scheme and `FDFiltered` gives exactly `FDNewton`'s root; with the paper's width it gives `1.854e-4` and `1.847e-4`, near it.
The Poisson root is a root of the filtered scheme too: Newton from it stops after one step, at `1.340e-4` and `0.587e-4` (rows `start`).
The published filtered values (`1.40e-4` at 63, `0.76e-4` at 127) lie just above that root and the standard column (`1.34e-4`, `0.59e-4`), so the paper's Newton reached a root near the convex one, from a start it does not state; ours reaches a non-convex one, with a `nonconvex_fraction` of 0.08 to 0.11 against 0.04 to 0.08 for the Poisson root (whose centred Hessian is indefinite in the flat disk as well, `monotone17.md`, Finding 6).
Level: mismatch, not tuned: choosing the start to land on the published root would make the result depend on knowing it.

## Finding 6: the cone

The monotone columns of Table 4.1 are not ours, for the reason of `monotone17.md`, Finding 4: with `f = 4/h^2` at the tip, the scheme's minimum over pairs at the tip of the exact cone is `2/h^2` or less, so the discrete cone is steeper than the exact one, and the error (`0.2929`, `0.691`, `1.566` with 9, 17 and 33 points) does not decrease with `N`.
The filtered scheme falls back to the monotone one at most nodes (`monotone_fraction` 0.4 to 0.96) and inherits it.
The construction of the problem is the paper's: the standard scheme on our cone matches Table 4.2's standard column at every printed `N`.
FO 2013 prints monotone values on the cone (`11.83e-3` with 9 points at `N = 31`) different from FO 2011's (`12e-3` to `11e-3`), so the 2011 and 2013 runs differ here too.

## Iterations and time

Our Newton takes 3 or 4 steps on C2 (published 2), 15 to 88 on T20 (2 to 6) and stops after 8 to 22 with 17 points on T2 (7 to 12); see Finding 4.
CPU times are 2013 MATLAB hardware and are not compared; at `N = 361` a filtered solve with 17 points takes 11 seconds on C2, 42 on T2, 214 on T20 and 527 on the cone, on one core of a100server.

## Not compared

- Table 4.2's hybrid columns: a second published table for `FDHybrid` (`hybrid.md`), on the centred example and with `n = N - 1`.
- Figures 4.2 and 4.3: Figure 4.3 shows where the filter chooses the monotone scheme; ours is `Result.info["monotone_fraction"]`, a share, not a map.
