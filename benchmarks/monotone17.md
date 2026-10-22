# Monotone wide-stencil scheme: published versus ours

Problems: T1 (smooth, `u = exp(|x|^2 / 2)`), T2 (corner blow-up, `u = -sqrt(2 - |x|^2)`), T20 (`C^1`, `u = ((|x - x0| - 0.2)^+)^2 / 2`, `f = 0` on a disk) and T21 (the cone `|x - x0|`, `f` a Dirac mass), on `(0, 1)^2` with Dirichlet data from the exact solution, `x0 = (0.5, 0.5)`.
Published numbers: Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 3, columns "9 Point", "17 Point" and "33 Point" (SINUM below), and Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Monotone", the 17-point stencil (JCP below).
Every value was checked against the page images; `sources.md` transcribes both tables with their conventions.

Ours: `FDMonotone(n=N, stencil=s)`, the preset `Newton(scheme=Monotone17(n=N, stencil=s), max_iter=200)` (ADR 0004), with the default `boundary="narrow"` unless a column says otherwise, residual tolerance `1e-10` relative to `max(1, max f)`, float64, scipy sparse LU.
Every run converged, except the T21 runs with `h = 1/N`, where `f` vanishes at every node.
Error: max over all grid nodes of `|u_h - u|`, `h = 1/N` (ADR 0001) unless a column says otherwise.
T21 is not in the catalogue (`Problem.f` cannot hold a measure); `run_monotone17.py` builds it per grid with the papers' `f = 4/h^2` at the nodes within `h/2` of `x0` and 0 elsewhere (SINUM, Section 6.1).
Raw data: `monotone17.csv` and `monotone17_iterates.csv`, written on a100server by `run_monotone17.py` (`just remote-bg monotone17 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/run_monotone17.py`), which logs every solve to MLflow under `maesolvers-monotone17-T1` and the like.

"As printed" rounds our value to the digits the paper prints; a bold value differs from the published one there.
The levels follow ADR 0002; the convergence order of a column is the least-squares slope of `log error` against `log N` over its printed `N`.

## Summary

| Problem | SINUM Table 3 (9, 17, 33 points) | JCP Table 3 (17 points) |
|---|---|---|
| T1 | order of magnitude; exact from `N = 127` on | order of magnitude; exact from `N = 181` on |
| T2 | exact | order of magnitude; exact from `N = 89` on, and at `N <= 63` with `h = 1/(N - 1)` (Finding 3) |
| T20 | 9 points order of magnitude; 17 and 33 points mismatch (Finding 5) | mismatch (Finding 5) |
| T21 | mismatch (Finding 4) | mismatch (Finding 4) |

`tests/test_reproduction.py` pins SINUM's T2 at `N = 31` and 63 and T1 at `N = 127`, all three stencils, to the printed digits; JCP's T2 at `N = 89` and 127 to the printed digits; and JCP's T1 at `N = 63` and 127 within `1e-3` relative.

## T1, smooth example

SINUM Table 3, in units of `1e-4`:

| N | 9 published | 9 ours | 9 as printed | 17 published | 17 ours | 17 as printed | 33 published | 33 ours | 33 as printed |
|---|---|---|---|---|---|---|---|---|---|
| 31 | 17.9 | 16.969 | **17.0** | 8.9 | 7.248 | **7.2** | 7.0 | 6.744 | **6.7** |
| 63 | 16.2 | 16.114 | **16.1** | 5.1 | 5.086 | 5.1 | 3.1 | 2.916 | **2.9** |
| 127 | 15.9 | 15.917 | 15.9 | 4.6 | 4.554 | 4.6 | 1.8 | 1.815 | 1.8 |
| 255 | 15.9 | 15.867 | 15.9 | 4.4 | 4.422 | 4.4 | 1.5 | 1.544 | 1.5 |
| 361 | 15.9 | 15.859 | 15.9 | 4.4 | 4.400 | 4.4 | 1.5 | 1.500 | 1.5 |

Ratio ours / published 0.95 to 1.00 (9 points), 0.81 to 1.01 (17), 0.94 to 1.03 (33); orders 0.04 against 0.02, 0.26 against 0.19, 0.62 against 0.59.

JCP Table 3, column "Monotone", in units of `1e-5`:

| N | Published | Ours | As printed | `boundary="interpolate"` | Newton steps |
|---|---|---|---|---|---|
| 31 | 89.09 | 72.477 | **72.48** | 64.818 | 5 |
| 45 | 60.50 | 57.560 | **57.56** | 54.331 | 6 |
| 63 | 50.88 | 50.865 | **50.86** | 49.331 | 7 |
| 89 | 47.51 | 47.349 | **47.35** | 46.637 | 8 |
| 127 | 45.53 | 45.539 | **45.54** | 45.209 | 11 |
| 181 | 44.65 | 44.650 | 44.65 | 44.495 | 12 |
| 255 | 44.22 | 44.222 | 44.22 | 44.146 | 13 |
| 361 | 44.00 | 44.003 | 44.00 | 43.966 | 14 |

Ratio 0.81 to 1.00, orders 0.24 against 0.18.
The error stops decreasing near `4.4e-4`: the 17-point stencil's angular resolution, not `h`, limits it, and likewise near `1.59e-3` for 9 points and `1.5e-4` for 33.

Level: order of magnitude for both tables.
From `N = 127` on (SINUM) and `N = 181` on (JCP) every value agrees to the printed digits, and at `N = 63` and 127 the JCP values agree to three of their four digits.
Below that the published errors are larger than ours, by up to 19 percent at `N = 31` (Finding 2).

## T2, example with blow-up

SINUM Table 3, in units of `1e-3`:

| N | 9 published | 9 ours | 17 published | 17 ours | 33 published | 33 ours |
|---|---|---|---|---|---|---|
| 31 | 1.7 | 1.655 | 1.7 | 1.655 | 1.7 | 1.655 |
| 63 | 0.9 | 0.863 | 0.6 | 0.572 | 0.6 | 0.572 |
| 127 | 0.8 | 0.834 | 0.3 | 0.348 | 0.2 | 0.203 |
| 255 | 0.8 | 0.826 | 0.3 | 0.329 | 0.2 | 0.161 |
| 361 | 0.8 | 0.825 | 0.3 | 0.326 | 0.2 | 0.154 |

Every value agrees with the published one at the printed digit, including the three equal values at `N = 31`, where the 9-point pairs attain the minimum at the node of largest error.
Level: exact.

JCP Table 3, column "Monotone", in units of `1e-3`:

| N | Published | Ours | As printed | `h = 1/(N - 1)` | `boundary="interpolate"` | Newton steps |
|---|---|---|---|---|---|---|
| 31 | 1.74 | 1.6547 | **1.65** | 1.7379 | 1.6547 | 10 |
| 45 | 0.98 | 0.9471 | **0.95** | 0.9795 | 0.9471 | 11 |
| 63 | 0.59 | 0.5721 | **0.57** | 0.5860 | 0.5721 | 13 |
| 89 | 0.37 | 0.3745 | 0.37 | 0.3758 | 0.3655 | 11 |
| 127 | 0.35 | 0.3485 | 0.35 | 0.3489 | 0.3445 | 15 |
| 181 | 0.34 | 0.3356 | 0.34 | 0.3358 | 0.3338 | 17 |
| 255 | 0.33 | 0.3294 | 0.33 | 0.3294 | 0.3285 | 14 |
| 361 | 0.33 | 0.3262 | 0.33 | 0.3262 | 0.3258 | 30 |

Ratio 0.95 to 1.01, orders 0.64 against 0.62.
Level: order of magnitude, and exact from `N = 89` on; at `N = 31` to 63 the published values are ours with `h = 1/(N - 1)` (Finding 3).
The centred Hessian of our node values has a negative eigenvalue at 1 to 10 nodes next to the corner (`nonconvex_fraction` 8e-5 to 1.5e-3), where it sees the blow-up; every second difference of the stencil is nonnegative there.

## T20, `C^1` example with a flat disk

SINUM Table 3, in units of `1e-3`:

| N | 9 published | 9 ours | 17 published | 17 ours | 33 published | 33 ours |
|---|---|---|---|---|---|---|
| 31 | 3.0 | 2.067 | 1.7 | 0.803 | 1.5 | 0.599 |
| 63 | 2.5 | 2.204 | 1.0 | 0.683 | 0.6 | 0.322 |
| 127 | 2.3 | 2.150 | 0.8 | 0.655 | 0.3 | 0.247 |
| 255 | 2.2 | 2.150 | 0.7 | 0.659 | 0.3 | 0.236 |
| 361 | 2.2 | 2.152 | 0.7 | 0.660 | 0.3 | 0.236 |

Ratio 0.69 to 0.98 (9 points), 0.47 to 0.94 (17), 0.40 to 0.82 (33); orders 0.12 against -0.01, 0.35 against 0.07, 0.64 against 0.36.

JCP Table 3, column "Monotone", in units of `1e-4`:

| N | Published | Ours | `h = 1/(N - 1)` | `boundary="interpolate"` | Newton steps |
|---|---|---|---|---|---|
| 31 | 17.5 | 8.03 | 8.70 | 7.98 | 10 |
| 45 | 11.6 | 6.78 | 6.84 | 6.76 | 7 |
| 63 | 9.8 | 6.83 | 6.82 | 6.82 | 9 |
| 89 | 8.4 | 6.64 | 6.51 | 6.64 | 18 |
| 127 | 7.9 | 6.55 | 6.63 | 6.55 | 19 |
| 181 | 7.4 | 6.55 | 6.61 | 6.55 | 28 |
| 255 | 7.2 | 6.59 | 6.58 | 6.59 | 48 |
| 361 | 7.0 | 6.60 | 6.62 | 6.60 | 49 |

Ratio 0.46 to 0.94, orders 0.33 against 0.06.
Level: order of magnitude for the 9-point column; mismatch for the 17- and 33-point columns of SINUM and for the JCP column, whose ratio falls below 1/2 at `N = 31` and whose orders differ by more than 0.2 (Finding 5).
The problem itself is the papers': the standard 9-point scheme under the Poisson iteration gives `2.57e-4`, `1.83e-4` and `1.45e-4` at `N = 31`, 45 and 63, the JCP column "Standard" `2.6e-4`, `1.8e-4` and `1.5e-4`.

## T21, the cone

With `h = 1/(N - 1)`, so that `x0` is a node, our error is the same at every `N`: `0.2929` with 9 points, `0.691` with 17 and `1.566` with 33, against the published `11e-3` to `12e-3`, `3e-3` to `4e-3` and `1e-3` to `3e-3`.
With `h = 1/N` (`N` odd) no node lies within `h/2` of `x0`, so `f = 0` at every node; Newton then stops at `max_iter` with errors near `0.48` (`N = 31`, 45).
Level: mismatch (Finding 4).

## Finding 1: the tables match the narrow stencil, not the boundary treatment the papers describe

Both papers say that where a wide stencil leaves the domain they "use interpolation at the boundary to construct a (lower accuracy) stencil" (SINUM p. 1700, JCP p. 823, after Oberman 2008, Section 3.1); issue #9 says the papers narrow the stencil next to the boundary instead.
We ran both: `boundary="interpolate"` shortens the arm to the boundary and takes the Dirichlet data there, `boundary="narrow"` leaves the wide pairs out where their arms do not end at a node.
On T1, where the 17- and 33-point columns differ most between the two, `"narrow"` agrees with the JCP column to four digits from `N = 181` on (44.65, 44.22, 44.00) and `"interpolate"` does not (44.50, 44.15, 43.97); with 33 points at `N = 63`, `"narrow"` gives `2.92e-4` and `"interpolate"` `2.54e-4` against the published `3.1e-4`.
So the published numbers come from the narrow stencil, whatever the text says, and `"narrow"` is the default.
The two agree with 9 points, whose arms always end at a node; on T2 and T20 they differ by at most 3 percent from `N = 63` on.

## Finding 2: at coarse grids the published T1 errors are larger than the converged ones

On T1 the published errors at `N = 31` to 63 are 4 to 23 percent above ours (all three stencils), and the gap closes by `N = 127` to 181.
The grid convention does not explain it: with `h = 1/(N - 1)` the 17-point error at `N = 31` is `7.44e-4`, still far below `8.909e-4`.
A Newton iterate stopped before convergence may: SINUM Table 1 reports 3 Newton steps at `N = 31` on T1, and Newton from the papers' start (`laplace u = sqrt(2 f)`, solved on the grid itself) gives, after 1 to 5 steps on the 17-point scheme (`monotone17_iterates.csv`):

| Steps | `N = 31` | `N = 45` | `N = 63` |
|---|---|---|---|
| 1 | 1.44e-2 | 3.58e-2 | 6.09e-2 |
| 2 | 1.42e-3 | 2.20e-3 | 4.41e-2 |
| 3 | 9.27e-4 | 9.30e-4 | 3.25e-2 |
| 4 | 8.07e-4 | 6.72e-4 | 1.05e-3 |
| 5 | 7.53e-4 | 5.95e-4 | 7.31e-4 |
| converged | 7.248e-4 | 5.756e-4 | 5.086e-4 |
| published (JCP) | 8.909e-4 | 6.050e-4 | 5.088e-4 |

The published value at `N = 31` lies between our third and fourth iterates, and at `N = 45` between the fourth and fifth.
The papers' start is solved on a coarse grid and interpolated, so their iterates are not ours, and they state no stopping rule; this makes an early stop plausible, not proven.

## Finding 3: the JCP T2 column at `N <= 63` matches `h = 1/(N - 1)`

At `N = 31`, 45 and 63 the JCP monotone values (`1.74e-3`, `0.98e-3`, `0.59e-3`) are ours with `n = N - 1` cells, to the printed digits (`1.7379e-3`, `0.9795e-3`, `0.5860e-3`), while at `N = 89` they are ours with `n = N` (`0.3745e-3`; with `n = N - 1`, `0.3758e-3` rounds to 0.38), and from `N = 127` on the two conventions round alike.
The JCP column "Hybrid" prints the same three values at `N = 31` to 63, so both columns there probably come from the same runs.
SINUM's coarser digits (1.7, 0.6) do not separate the two conventions.
For the standard scheme every printed value of both tables follows `h = 1/N` (`standard_fd.md`, Finding 1); why the coarse monotone runs on T2 would use another grid is not known.
It does not carry over to T1 or T20 (Finding 2, Finding 5).

## Finding 4: the cone's discrete Dirac mass does not fit the scheme

The papers replace the Dirac mass `pi delta_x0` by `4/h^2` at `x0`.
At the tip of the exact cone the axis pair gives `(2/h)(2/h) = 4/h^2`, but the diagonal pair gives `(sqrt 2/h)^2 = 2/h^2`, and the wider pairs less; the scheme takes the minimum, so the exact cone sampled at the nodes is a subsolution at the tip, and the discrete solution is a steeper cone.
Its error is a constant independent of `h` (`0.2929 = (sqrt 2 - 1)/sqrt 2` with 9 points), so the scheme with this `f` does not converge to the cone at all, and the published errors near `1e-3` to `1e-2` cannot come from it.
The papers do not say how they discretised the mass otherwise.
With `h = 1/N` and `N` odd, the papers' rule puts no mass on any node.
Not reproduced; the catalogue keeps T21 out.

## Finding 5: on T20 the published errors stay above the convex discrete solution's

Where `f = 0`, Froese and Oberman's scheme `min (D u)^+ (D' u)^+ = f` has more than one root: any grid function with a negative second difference in each pair solves it there.
Their Newton iteration, from their start and with their stopping rule, reaches one of them; ours reaches the discretely convex one, which `Monotone17` makes the only root (`docs/methods/monotone17.md`, "Below zero").
In scouting runs (not receipts) with the bare positive-part residual and 33 points at `N = 31`, two starts converged to roots with errors near `8e-4` and `1.8e-3`, on either side of the published `1.5e-3`.
Ours is lower than published at every `N`, by a factor 2.2 at `N = 31` falling to 1.06 at `N = 361` (JCP), and our error is flat in `N` where theirs decreases.
The grid convention does not explain it: with `h = 1/(N - 1)` the 17-point error at `N = 31` is `8.70e-4`.
The published values may come from a non-convex root, or from Newton stopped early in the flat disk, where the papers' Jacobian floor makes Newton converge only linearly (`docs/methods/monotone17.md`, "Newton's method"); the tables do not let us tell which.
The level is therefore mismatch, recorded and not tuned: the convex root is the one the theory (SINUM, Theorem 6) is about.

## Iterations and time

Newton needs 5 to 14 steps on T1 with 17 points (SINUM Table 1: 3 to 7), 10 to 30 on T2 (4 to 9) and 7 to 49 on T20 (4 to 20), from a different start and with a stopping rule the papers do not state.
With the papers' Jacobian floor `1e-8 / (2 h^2)` Newton converges only linearly on T20: 417 steps at `N = 181` against 28 with the default floor `1e-12 / (2 h^2)`, at the same root (`tests/test_monotone.py` checks that the floor does not change the solution).
CPU times are 2011 MATLAB hardware and are not compared.

## Not compared

- SINUM Table 4 (3D, 19-point stencil): 3D boxes come with 0.2.
- BFO 2010, Table 7, the 9-, 17- and 33-point columns on T7: they are the wide-stencil scheme of Oberman 2008, the product of the smallest and largest directional second difference, not the scheme of the 2011 papers.
- The JCP column "Hybrid": a separate scheme, planned for 0.1.
