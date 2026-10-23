# Filtered scheme: published versus ours

Problems (Froese-Oberman 2013, Section 4.2, on `[0, 1]^2` with Dirichlet data from the exact solution, `x0 = (0.5, 0.5)`): the `C^2` example `u = exp(|x - x0|^2 / 2)`, which is T1 centred at `x0` and not in the catalogue (called C2 below, `C2-centred` in the receipts; Finding 1); T20 (`C^1`, `f = 0` on a disk); T2 (corner blow-up, uncentred); and the cone T21 with `f = 4/h^2` at the nodes within `h/2` of `x0`, built per grid by `run_monotone17.cone`.
Published numbers: Froese-Oberman, SIAM J. Numer. Anal. 51 (2013), Table 4.1 (monotone and filtered, 9, 17 and 33 points) and Table 4.2 (17 points: filtered, its Newton iterations, and the standard scheme); `sources.md`, section 5, transcribes both, checked against the page images.
The hybrid columns of Table 4.2 belong to the hybrid scheme (`hybrid.md`) and are not compared here; `sources.md`, section 5, transcribes them.

Ours, with `n = N - 1` cells (Finding 1):

- filtered: `FDFiltered(n, stencil)`, the preset `Newton(scheme=Filtered(accurate=Standard9(n), monotone=Monotone17(n, stencil)), max_iter=200)`, filter width `sqrt(h) + dtheta / 10` with `dtheta` the largest angle between neighbouring stencil directions (Finding 3, `docs/methods/filtered.md`);
- monotone: `FDMonotone(n, stencil)` (`monotone17.md`);
- standard: `FDPoisson(n)`, the standard 9-point scheme under the Poisson iteration, as the paper's "one of the methods of [BFO10]".

All with `boundary="narrow"` unless a row says otherwise, residual tolerance `1e-10` relative to `max(1, max f)` for Newton, float64, scipy sparse LU.
Error: max over all grid nodes of `|u_h - u|`.
Conventions that differ from the paper's: the monotone part is `Monotone17`, continued below zero with the slope `kappa` (`monotone17.md`), where the paper's operator adds `sum min{D, delta}`; and the paper "regularized the max operator to allow the gradient to be computed, as in [FO11b]" (p. 438), with no `delta` printed, where `Monotone17` takes the bare max and min with their semi-smooth Jacobian.
`filtered_checks.csv` runs both alternatives (rows `kappa` and `regularised`, the T2 section and Finding 4).
Raw data: `filtered.csv`, written on a100server by `run_filtered.py` at commit e9ddc67 (tmux session `maesolvers-filtered4`, `run_filtered.py --processes 16` with one thread per process), and `filtered_checks.csv` by `run_filtered.py --checks --processes 16` at commit dd86994 (session `maesolvers-filtered5-checks`).
Every solve is one run in the MLflow experiment `maesolvers-filtered-<problem>` (`C2-centred`, T20, T2, T21, T1, and T3, T4, T5, T7 for the checks), tagged with that commit and a clean checkout; runs of earlier generations, with the half-gap width, carry older commits.

A bold value differs from the published one at the printed digits; a star marks a solve that did not converge (Finding 4).
A level read off starred values is provisional: it depends on where `Newton`'s line search gave up.
For every unconverged filtered solve `filtered.csv` gives, next to the last iterate's error, the error of the iterate eight steps before (`max_err_8_steps_before`), so a reader can judge how settled each starred value is.
The levels follow ADR 0002; the order of a column is the least-squares slope of `log error` against `log N` over the printed `N`.

## Summary

| Problem | Table 4.1 monotone | Table 4.1 filtered | Table 4.2 filtered | Table 4.2 standard |
|---|---|---|---|---|
| C2 | 17 points exact; 9 and 33 order of magnitude (exact from `N = 63`) | order of magnitude (exact but at `N = 63`, Finding 2) | exact | exact |
| T20 | order of magnitude (last digit) | mismatch (Finding 5) | mismatch (Finding 5) | exact |
| T2 | exact | mismatch: no solve converges (Finding 4); the last iterates agree to the printed digits in 14 of 15 cells | mismatch: not converged; the last iterates agree in 4 of 5 cells | exact |
| T21 | mismatch (Finding 6) | mismatch | mismatch | exact |

`tests/test_reproduction.py` pins the filtered values on C2 at `N = 31` (all three stencils) and at `N = 63` (17 points, Table 4.2), the standard values on C2, T2 and T20, and monotone values on C2 and T2, each to the printed digits, all with `n = N - 1`.
No T2 filtered value is pinned: none is a converged solve.

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
| 31 | 21.54 | **21.64** | 3.73 | **3.69** | 8.66 | **8.70** | 3.99 | **3.71** | 6.39 | 6.39 | 3.67 | **3.72** |
| 63 | 21.33 | **21.32** | 1.51 | **1.73** | 6.82 | 6.82 | 1.40 | **1.73** | 3.18 | **3.17** | 1.46 | **1.77** |
| 127 | 21.55 | **21.54** | 0.92 | **2.01*** | 6.63 | 6.63 | 0.76 | **1.72*** | 2.49 | 2.49 | 0.78 | **1.62*** |
| 255 | 21.51 | **21.50** | 0.38 | **0.74*** | 6.58 | 6.58 | 0.46 | **0.67*** | 2.36 | 2.36 | 0.37 | **3.51*** |
| 361 | 21.53 | 21.53 | 0.23 | **2.83*** | 6.62 | 6.62 | 0.31 | **2.63*** | 2.37 | 2.37 | 0.28 | **0.47*** |

Table 4.2, in units of `1e-4`:

| N | Filtered | ours | Newton steps, published | ours | Standard | ours |
|---|---|---|---|---|---|---|
| 31 | 3.99 | **3.71** | 2 | 12 | 3.78 | 3.78 |
| 63 | 1.40 | **1.73** | 2 | 16 | 1.34 | 1.34 |
| 127 | 0.76 | **1.72*** | 3 | 27 | 0.59 | 0.59 |
| 255 | 0.46 | **0.67*** | 3 | 42 | -- | |
| 361 | 0.31 | **2.63*** | 6 | 75 | -- | |

Monotone: ratio 1.00 in every column, orders equal to two decimals; the 9-point column differs in the last printed digit at four of five `N`, the others at one.
These are the convex roots of `Monotone17`, and they match where FO 2011's monotone columns on the same problem did not (`monotone17.md`, Finding 5): the 2013 numbers support that finding's reading that the 2011 ones came from another root or another scheme.
Filtered: ratio 0.93 to 12; orders published 1.10, 0.99, 1.04, ours 0.27, 0.32, 0.49; mismatch (Finding 5).
From `N = 127` on no solve converges (Finding 4), and the starred values move: over the last eight steps by less than `1e-5` relative in all cells but one (9 points at `N = 361`, `1e-4`), so they are where the line search stopped, not drifting iterates.

## T2, the example with blow-up

Table 4.1, in units of `1e-3`:

| N | 9 monotone | ours | 9 filtered | ours | 17 monotone | ours | 17 filtered | ours | 33 monotone | ours | 33 filtered | ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 1.74 | 1.74 | 1.74 | 1.74* | 1.74 | 1.74 | 1.74 | 1.74* | 1.74 | 1.74 | 1.74 | 1.74* |
| 63 | 0.86 | 0.86 | 0.59 | 0.59* | 0.59 | 0.59 | 0.59 | 0.59* | 0.59 | 0.59 | 0.59 | 0.59* |
| 127 | 0.83 | 0.83 | 0.20 | 0.20* | 0.35 | 0.35 | 0.20 | 0.20* | 0.20 | 0.20 | 0.20 | 0.20* |
| 255 | 0.83 | 0.83 | 0.15 | 0.15* | 0.33 | 0.33 | 0.13 | **0.12*** | 0.16 | 0.16 | 0.08 | 0.08* |
| 361 | 0.83 | 0.83 | 0.17 | 0.17* | 0.33 | 0.33 | 0.13 | 0.13* | 0.15 | 0.15 | 0.08 | 0.08* |

Table 4.2, in units of `1e-3`:

| N | Filtered | ours | Newton steps, published | ours | Standard | ours |
|---|---|---|---|---|---|---|
| 31 | 1.74 | 1.74* | 7 | 22 | 17.38 | 17.38 |
| 63 | 0.59 | 0.59* | 9 | 26 | 12.62 | 12.62 |
| 127 | 0.20 | 0.20* | 11 | 11 | 9.04 | 9.04 |
| 255 | 0.13 | **0.12*** | 8 | 19 | 6.43 | 6.43 |
| 361 | 0.13 | 0.13* | 12 | 200 | 5.42 | 5.42 |

Monotone: exact, all three stencils.
Filtered, with `boundary="narrow"`: no solve converges (Finding 4); every one ends with a max residual between `1e-2` and `2e-1`, against a tolerance of `1e-10 max(1, max f)`.
The last iterates agree with the published values to the printed digits in 14 of the 15 cells of Table 4.1 (17 points at `N = 255`: `0.121e-3` against `0.13e-3`), with ratio 0.93 to 1.02 and orders 0.98, 1.09, 1.31 against the published 0.99, 1.09, 1.32.
That would be exact for 9 and 33 points and order of magnitude for 17, but it is not a reproduction: the values are not discrete solutions, and the paper reports 7 to 12 Newton steps, so its solves converged.
How settled the starred values are differs by cell (`max_err_8_steps_before`): with 17 points at `N = 31`, 63 and 361 and with 33 points at 31 to 127 the error changed by less than `1e-4` relative over the last eight steps; in the other nine cells by 2 percent or more, and in short solves by many times the final error (17 points at `N = 127` stopped after 11 steps, and eight steps before its error was 19 times larger).
Part of the agreement does not discriminate: at `N = 31` and 63 every filtered solve but one (below) ends at the same error, `1.7379e-3` and `5.8602e-4`, which is also the narrow 17- and 33-point monotone scheme's, and so do all widths, starts and `kappa` in `filtered_checks.csv`; the max error sits at nodes near the corner where all these schemes coincide.
With `boundary="interpolate"` one solve converges: 33 points at `N = 31`, error `1.011e-3` against the published `1.74e-3`, which is the 33-point monotone scheme's with that boundary (`1.012e-3`); the other nine do not converge, and their last iterates print as the published values.
The paper notes that with its `eps` the filtered scheme "is slightly less accurate on the blow-up example".
The filter width decides the large-`N` cells: with half the angular gap as `dtheta` (Oberman's angular resolution) the last iterates at `N = 255` and 361 are `0.25e-3` and `0.28e-3` with 9 points, `0.15e-3` and `0.16e-3` with 17 and `0.088e-3` and `0.092e-3` with 33 (`filtered_checks.csv`, rows `width`), against the published `0.15`, `0.17`, `0.13`, `0.13`, `0.08`, `0.08` (Finding 3).
`Monotone17`'s slope below zero (`kappa = sqrt(max(f, mean f, 1))`, large near the corner, where the paper's operator has slope 1) changes the error by less than `1e-6` relative at `N = 63` and 127, and at 255 gives `0.125e-3` against `0.121e-3` (rows `kappa`), the published value but not converged either.

## T21, the cone

Table 4.1, in units of `1e-3`:

| N | 9 monotone | ours | 9 filtered | ours | 17 monotone | ours | 17 filtered | ours | 33 monotone | ours | 33 filtered | ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 11.83 | **292.89** | 10.42 | **292.89*** | 3.56 | **690.60** | 4.16 | **690.41*** | 1.61 | **1565.74** | 1.23 | **1564.62*** |
| 63 | 11.10 | **292.89** | 11.56 | **292.89*** | 3.49 | **690.90** | 2.82 | **477.98*** | 1.65 | **1565.71** | 1.90 | **1565.73*** |
| 127 | 11.80 | **292.89** | 10.97 | **259.59*** | 3.45 | **690.96** | 2.83 | **690.38*** | 1.64 | **1565.74** | 1.11 | **1565.31*** |
| 255 | 10.47 | **292.89** | 11.03 | **117.09*** | 3.46 | **690.98** | 3.06 | **431.61*** | 1.64 | **1565.74** | 1.15 | **1352.02*** |
| 361 | 10.40 | **292.89** | 10.37 | **92.71*** | 3.45 | **690.98** | 3.22 | **469.70*** | 1.64 | **1565.74** | 1.12 | **1480.96*** |

Table 4.2, standard column, in units of `1e-3`: published 5.19, 2.86, 1.54 at `N = 31`, 63, 127, ours 5.19, 2.86, 1.54.
Mismatch for the monotone and the filtered schemes (Finding 6); exact for the standard scheme.

## Finding 1: the `C^2` example is centred, and `N` counts nodes

Eq. (4.1) prints `u = exp(|x - x0|^2 / 2)` with `x0 = (0.5, 0.5)` (checked on the page image), although the text calls the four examples those of FO 2011, whose `C^2` example is T1, `exp(|x|^2 / 2)`.
And the paper's `N x N` grid has `N` nodes per side: with `n = N - 1` cells every value of the Table 4.2 standard column is ours to the printed digits, on all four problems (C2: 4.5406, 1.0641, 0.2577, 0.0634, 0.0316 in units of `1e-5`; T20: 3.785, 1.340, 0.587 in `1e-4`; T2: 17.384, 12.623, 9.038, 6.430, 5.417 in `1e-3`; T21: 5.192, 2.860, 1.542 in `1e-3`), and with `n = N` none is (C2 at `N = 31`: `4.246e-5`; T2: `17.15e-3`, the FO 2011 value; `filtered.csv`, rows `1/N`).
The monotone columns agree: on T2, `h = 1/(N - 1)` is the convention that the JCP 2011 monotone column follows at `N <= 63` (`monotone17.md`, Finding 3), and here it fits at every `N`.
This explains the disagreement noted in `papers.md` and the survey: Table 4.2's standard column (`4.54e-5` at `N = 31` on the "C^2 example", `17.38e-3` on T2) differs from FO 2011's (`7.14e-5`, `17.15e-3`) because it is another problem on another grid, not another discretisation; the pinned FO 2011 values stand.
The centred example has no T id; `tests/centred.py` defines it for the tests, and `run_filtered.py` builds the same problem.

## Finding 2: Table 4.1 and Table 4.2 print different filtered values at `N = 63` on C2

Table 4.1 prints `1.04e-5` in all three filtered columns at `N = 63`; Table 4.2 prints `1.06e-5` for the same 17-point filtered solve, and `1.06e-5` for the standard scheme.
The filter keeps the standard scheme at every node there, so the filtered and standard values must agree; ours is `1.0641e-5` for both.
We read `1.04` as a misprint and compare the filtered column with Table 4.2; it is the only cell of Table 4.1 that disagrees with Table 4.2.

## Finding 3: `dtheta` is the full angular gap of the stencil

The paper takes `eps = sqrt(h) + dtheta / 10` and gives `dtheta` no value, here or in FO 2011 (SINUM p. 1700 names it the parameter of the directional discretisation).
Oberman 2008, eq. (3.3), defines the directional resolution as `max over theta of min over i |theta - theta_i|`, half the largest gap between stencil directions (`pi/8`, `atan(1/2)/2`, `atan(1/3)/2`; `angular_resolution`).
The paper's T2 columns fit the full gap (`pi/4`, `atan(1/2)`, `atan(1/3)`) instead, so `Filtered` takes `dtheta = 2 * angular_resolution`:

| N | stencil | published | full gap (default) | half gap |
|---|---|---|---|---|
| 255 | 9 | 0.15 | 0.147* | 0.252* |
| 361 | 9 | 0.17 | 0.173* | 0.278* |
| 255 | 17 | 0.13 | 0.121* | 0.148* |
| 361 | 17 | 0.13 | 0.134* | 0.160* |
| 255 | 33 | 0.08 | 0.081* | 0.088* |
| 361 | 33 | 0.08 | 0.081* | 0.092* |

T2 filtered, units of `1e-3` (`filtered_checks.csv`, rows `width`); with 17 points at `N = 63` and 127 the two widths give the same errors to six digits.
None of these solves converges (Finding 4), so the evidence is the last iterates'; with the half gap (the `filtered.csv` of commit 4130727) the 9-point column has order 0.75 against the published 0.99, with the full gap 0.98.
On C2 the width does not matter: the filter keeps the standard scheme everywhere with either.
On T20 the full gap moves the error away from the published one rather than towards it: `1.733e-4` at `N = 63` against `1.854e-4` with the half gap and the published `1.40e-4`; see Finding 5.

## Finding 4: Newton with the paper's Jacobian does not converge

In the band `1 < |F_A - F_M| / eps < 2` the filtered equation is `2 F_M - F_A + 2 eps sign(F_A - F_M)`, whose derivative is `2 J_M - J_A`; the paper takes `2 J_M`, dropping `-J_A`, because a negative weight on `J_A` "can lead to poorly conditioned or ill-posed linear systems" (p. 438).
Our `Newton` takes that Jacobian, halves the step until the max residual decreases (down to `2^-39`), and ends the solve, not converged, when no step does.
It ends that way on T2 at every `N`, stencil and boundary treatment but one (33 points, `boundary="interpolate"`, `N = 31`, see the T2 section), on T20 from `N = 127` on, on the cone at every `N`, and on T1 (below) at every `N` but one; on C2, where no node is in the band, and on T20 at `N <= 63`, Newton converges in 3 to 21 steps.
The paper reports 2 to 12 Newton steps with the same approximate Jacobian, on T2 7 to 12, so its solves converged where ours do not.

Three situations stop the line search, on different problems:

- A node in the band attains the max residual, and the Newton direction, which there is not the derivative's, need not decrease it: on T2 (at `n = 62` and 126) and T1 (at `n = 64` and 127), with the residual between `1e-2` and `2e-1`; on T20 at `n = 126` the max sits at a node of the accurate piece and band nodes hold the next four (an exploratory diagnosis, not a receipt).
- The approximate Jacobian is nearly singular, so no step decreases the residual at all: on T5 at `n = 16` the solve takes 0 steps and returns the Laplace start (`filtered_checks.csv`, rows `catalogue`).
  In an exploratory diagnosis (not a receipt): at the start the max residual sits at a node in the monotone piece (`|x| = 1.04`, where `f = 4`), with 139, 67 and 19 nodes in the accurate piece, the band and the monotone piece; the approximate Jacobian has condition number `5e13` against `1.2e3` for the exact one (`2 J_M - J_A` in the band), and the Newton direction has entries near `6e10`.
  On the cone at `n = 62` the approximate Jacobian is exactly singular after one step (scipy warns), and the solve stops after 2 steps.
- On T4, whose `f` jumps from 0 to 1 on the circle `|x - x0| = 0.4`, the solve stops after 25 to 30 steps with the max residual at a node of the accurate piece next to that circle (`|x - x0| = 0.375` at `n = 16`, `0.399` at `n = 64`, same diagnosis): the residual is not smooth there, and neither mechanism above applies. On the cone at `n = 126` the max residual also sits at a node of the accurate piece when the solve stops, after 103 steps.

Remedies that do not converge either:

- the paper's own device, the regularised max of FO 2011 (SINUM Section 3.5, `run_monotone17.SmoothMonotone17`) in the monotone part, with `delta = 1e-4`, `1e-2` and `0.1` (rows `regularised`): on T2 at `N = 63` and 127 no `delta` converges, `1e-4` and `1e-2` diverge to errors of `0.02` to `0.66`, and `0.1` ends at `5.864e-4` at 63 (the bare scheme's `5.860e-4`) and diverges at 127; on T1 one of six solves converges (`delta = 1e-2`, `N = 63`, 156 steps, `4.55e-5`); on T20 all three converge at `N = 63`, to the bare scheme's root, and only `delta = 1e-4` at 127, to `FDNewton`'s root;
- starts from the monotone or the standard root (rows `start`): on T2 Newton stalls from both, after 1 to 13 steps, at the same errors; on T20 the monotone start stalls, while the standard root is already a filtered root (Finding 5);
- other widths (rows `width`): twice and four times the width converge on T20 at `N = 127`, not on T2;
- `kappa = 1` in the monotone part (rows `kappa`);
- in exploratory runs not kept as receipts, with the half-gap width: undamped steps converge linearly, with ratio about 0.87, on T2 at `n = 30`, and end with residuals of `4e2` and `4e3` after 300 steps at `n = 126` and 254; with the exact derivative `2 J_M - J_A` the steps diverge from the start.

What the paper did differently is not stated: it gives no start, no damping, no stopping rule and no `delta`.

On the catalogue problems `FDFiltered` supports, with the default width and `max_iter = 200` (rows `catalogue`):

| Problem | `n = 16` | `n = 64` |
|---|---|---|
| T1 | converged, 73 steps | not converged, residual `1e-2` |
| T2 | converged, 119 steps | not converged in 200 steps, residual `4e-2` |
| T3 | converged, 94 steps | converged, 68 steps |
| T4 | not converged, residual `0.14` | not converged, residual `0.99` |
| T5 | not converged, 0 steps, residual `6.6` | not converged, residual `0.44` |
| T7 | converged, 32 steps | not converged, residual `5e-2` |
| T20 | converged, 12 steps | not converged, residual `3e-3` |

On T1 itself (uncentred, `n = N`), which the paper does not run, `FDFiltered` converges once, with 9 points at `N = 63`; the last iterates have errors between `4.4e-5` and `1.4e-4` with 17 points and between `8.3e-6` and `1.5e-4` with 33 at `N = 31` to 361, mostly between the monotone scheme's (`4.4e-4` and `1.5e-4` at `N = 361`) and the standard one's (`7.1e-5` at `N = 31` to `5.3e-7` at 361), and between `1.8e-4` and `7.2e-4` with 9 points; 0.8 to 10 percent of the nodes, near the corner `(1, 1)` where the Hessian is most anisotropic, leave the standard scheme (`filtered.csv`, rows `T1`).

## Finding 5: on T20 the filtered scheme inherits the standard scheme's non-convex root

Where the filter keeps the standard scheme the filtered equation is the standard one, and Newton on the standard 9-point scheme, from our start, reaches a non-convex discrete root on T20, as it does on T2 (`standard_fd.md`, Finding 2): `FDNewton` gives `1.733e-4` at `N = 63` and `1.992e-4` at 127, the convex root of the Poisson iteration `1.340e-4` and `0.587e-4` (`filtered_checks.csv`, rows `newton` and `start`).
With the default width the filter keeps the standard scheme at every node at `N = 63` with 9 and 17 points, and `FDFiltered` gives exactly `FDNewton`'s root, `1.7327e-4`; with 33 points `1.770e-4`.
At `N = 127` the default width does not converge (`1.722e-4`, Finding 4); four times the width gives exactly `FDNewton`'s root, `1.9922e-4`, and twice the width another root of the standard scheme, `1.9944e-4`, with no node on the monotone scheme (rows `width`).
The Poisson root is a root of the filtered scheme too: Newton from it stops after one step, at `1.340e-4` and `0.587e-4` (rows `start`).
The published filtered values (`1.40e-4` at 63, `0.76e-4` at 127) lie just above that root and the standard column (`1.34e-4`, `0.59e-4`), so the paper's Newton reached a root near the convex one, from a start it does not state; ours reaches a non-convex one, with a `nonconvex_fraction` of 0.07 to 0.11 against 0.04 to 0.08 for the Poisson root (whose centred Hessian is indefinite in the flat disk as well, `monotone17.md`, Finding 6).
Level: mismatch, not tuned: choosing the start to land on the published root would make the result depend on knowing it.

## Finding 6: the cone

The monotone columns of Table 4.1 are not ours, for the reason of `monotone17.md`, Finding 4: with `f = 4/h^2` at the tip, the scheme's minimum over pairs at the tip of the exact cone is `2/h^2` or less, so the discrete cone is steeper than the exact one, and the error (`0.2929`, `0.691`, `1.566` with 9, 17 and 33 points) does not decrease with `N`.
The filtered scheme falls back to the monotone one at most nodes (`monotone_fraction` 0.46 to 0.97 at the last iterate; no solve converges) and inherits it.
The construction of the problem is the paper's: the standard scheme on our cone matches Table 4.2's standard column at every printed `N`.
FO 2013 prints monotone values on the cone (`11.83e-3` with 9 points at `N = 31`) different from FO 2011's (`12e-3` to `11e-3`), so the 2011 and 2013 runs differ here too.

## Iterations and time

With 17 points our Newton takes 3 or 4 steps on C2 (published 2), 12 to 75 on T20 (2 to 6) and stops after 11 to 26 steps on T2, or runs out of its 200 at `N = 361` (published 7 to 12); see Finding 4.
CPU times are 2013 MATLAB hardware and are not compared; at `N = 361` a filtered solve with 17 points takes 11 seconds on C2, 400 on T2 (200 steps), 167 on T20 and 15 on the cone (3 steps), on one core of a100server.
`Filtered.jacobian` evaluates both parts' residuals again at the iterate where `residual` has just done so, one more residual evaluation per step, kept so that the scheme holds no state.

## Not compared

- Table 4.2's hybrid columns: a second published table for `FDHybrid` (`hybrid.md`), on the centred example and with `n = N - 1`.
- Figures 4.2 and 4.3: Figure 4.3 shows where the filter chooses the monotone scheme; ours is `Result.info["monotone_fraction"]`, a share, not a map.
