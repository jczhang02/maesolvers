# Multigrid: published versus ours

Problems, with Dirichlet data from the exact solution: in 2D on `(0, 1)^2`, T1 (`u = exp(|x|^2 / 2)`, Example 4.1), T3 (`u = (2 sqrt 2 / 3) |x|^(3/2)`, Example 4.2) and T2 (`u = -sqrt(2 - |x|^2)`, Example 4.3); in 3D, T9 (`u = exp(|x|^2 / 2)` on `(0, 1)^3`, Example 4.4), T46 (`u = |x|^2 / 2 - sin x - sin y - sin z` on `(0, pi)^3`, Example 4.5), T28 (`u = |x|^(3/2) / 3` on `(0, 1)^3`, Example 4.6) and T22 (`u = -sqrt(3 - |x|^2)` on `(0, 1)^3`, Example 4.7).
Published numbers: J. Liu, B. D. Froese, A. M. Oberman, M. Xiao, Int. J. Comput. Math. 94 (2017) 1850-1866, Tables 1 and 2 (2D, red-black and lexicographic Gauss-Seidel) and Tables 3, 5, 6 and 7 (3D, lexicographic), each checked against the page images; the text layer of the PDF garbles the first rows of Tables 5 to 7 and prints Example 4.5's `|x|^2 / 2` as `(x + y + z) / 2`.
Table 4 reprints numbers of other papers and is not a target (see "Not compared").

Ours: `FDMultigrid(n=N, ordering=...)`, the preset `Multigrid(scheme=StandardBox(n=N))` (ADR 0004; `docs/methods/multigrid.md`), with the paper's settings, which are the defaults: one FMG cycle, then FAS V-cycles with 2 + 2 nonlinear Gauss-Seidel sweeps and one sweep on the grid of 2 cells per side, until the relative residual is at most `1e-6`; float64.
Every run of `multigrid.csv` converged.
Error: max over all grid nodes of `|u_h - u|`, `evaluate`'s `max`, `h = 1/N` (ADR 0001; `pi/N` on T46).
"Converged" is the error of the discrete solution itself: the same preset (red-black) for ten cycles, the FMG cycle and nine V-cycles, whatever the relative residual, which is then at most `3e-16` (except on T2, `1.4e-11`, and T22 at `N = 128`, Finding 4).
The Gauss-Seidel columns repeat the paper's stand-alone solver: lexicographic sweeps of `StandardBox.relax` from `1.01 u` at the unknowns until the residual's discrete L2 norm is `1e-6` times that of the start.
Raw data: `multigrid.csv` (82 rows), written on a100server by `run_multigrid.py` at commit 64fe66a, clean checkout (`just remote-bg multigrid-r2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/run_multigrid.py --workers 6 --out /jczhang02/maesolvers-data/data/multigrid.csv`, then copied here), and `multigrid_checks.csv` by `run_multigrid.py --checks --workers 12` at commit f11539d, whose library code is the same (its eight T2 rows with the paper's root selection repeat `multigrid.csv` to all digits).
An earlier run at 85bcd7c gave the same values in every column but the times and the converged error, which it took with `tol=1e-12`; that tolerance stopped after the FMG cycle on the larger grids, where the reference of the relative residual is large (Finding 2).
Every row is one run in the MLflow experiment `maesolvers-multigrid-<problem>` (T1, T2, T3, T9, T22, T28, T46), named `table<k>-<ordering>-n<N>`, `check-<root>-<ordering>-n<N>` or `check-smallest-<ordering>-n<N>-cycles<k>`, tagged with its commit and a clean checkout; the experiments also hold the earlier generation (85bcd7c, and the T2 checks at fe47d9b).
`run_multigrid.py --report` prints the tables below from the CSV.

"As printed" rounds our value to the digits the paper prints; a bold value differs from the published one there.
The levels follow ADR 0002; the convergence order of a column is the least-squares slope of `log error` against `log N` over its printed `N`, written "orders: published X, ours Y"; a ratio is ours divided by published.

## Summary

| Table | Problem | FMG-FAS | Gauss-Seidel |
|---|---|---|---|
| 1 (red-black) | T1, T3, T2 | order of magnitude (ratios 0.94 to 1.08, orders equal to 0.02; "Iter" equal in all 21 cells) | |
| 2 (lexicographic) | T1, T3, T2 | order of magnitude (ratios 0.92 to 1.05, orders equal to 0.01; "Iter" equal in 19 of 21 cells) | |
| 3 | T9 | order of magnitude (ratios 1.14 to 1.99; Finding 1) | exact; same sweeps |
| 5 | T46 | order of magnitude (ratios 1.09 to 1.60; Finding 1) | exact; same sweeps |
| 6 | T28 | order of magnitude (ratios 0.80 to 0.84) | exact; same sweeps at `N = 8`, 16, one fewer at 32 |
| 7 | T22 | mismatch at `N = 32` (ratio 0.25; Finding 4), order of magnitude at every other `N` | exact; same sweeps at `N = 8`, 16, three fewer at 32 |

`tests/test_reproduction.py` pins the Gauss-Seidel errors to the printed digits and the sweep counts at `N = 8` on the four 3D problems and at `N = 16` on T9.

## Tables 1 and 2: 2D

Max error, red-black (Table 1):

| N | T1 published | T1 ours | T3 published | T3 ours | T2 published | T2 ours | T2 Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| 128 | 4.5e-06 | 4.447e-06 | 3.8e-05 | 3.626e-05 | 8.9e-03 | 8.959e-03 | 2, 2 |
| 256 | 1.1e-06 | 1.111e-06 | 1.3e-05 | 1.283e-05 | 6.4e-03 | 6.396e-03 | 2, 2 |
| 512 | 2.8e-07 | 2.778e-07 | 4.7e-06 | 4.535e-06 | 4.5e-03 | 4.521e-03 | 1, 1 |
| 1024 | 7.0e-08 | 6.944e-08 | 1.7e-06 | 1.603e-06 | 3.2e-03 | 3.204e-03 | 1, 1 |
| 2048 | 1.7e-08 | 1.736e-08 | 5.9e-07 | 5.669e-07 | 2.3e-03 | 2.268e-03 | 1, 1 |
| 4096 | 4.3e-09 | 4.340e-09 | 2.1e-07 | 2.004e-07 | 1.6e-03 | 1.605e-03 | 1, 1 |
| 8192 | 1.0e-09 | 1.082e-09 | 7.4e-08 | 7.087e-08 | 1.1e-03 | 1.135e-03 | 1, 1 |

Ratios 0.99 to 1.08 (T1), 0.94 to 0.99 (T3), 0.99 to 1.03 (T2); orders: published 2.02, 1.50, 0.50, ours 2.00, 1.50, 0.50.
T1 and T3 take one cycle at every `N` in both tables, as published.

Max error, lexicographic (Table 2):

| N | T1 published | T1 ours | T3 published | T3 ours | T2 published | T2 ours | T2 Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| 128 | 5.1e-06 | 5.357e-06 | 4.3e-05 | 4.018e-05 | 8.9e-03 | 8.961e-03 | 3, 2 |
| 256 | 1.3e-06 | 1.341e-06 | 1.5e-05 | 1.422e-05 | 6.3e-03 | 6.399e-03 | 2, 2 |
| 512 | 3.3e-07 | 3.355e-07 | 5.4e-06 | 5.029e-06 | 4.4e-03 | 4.547e-03 | 1, 2 |
| 1024 | 8.2e-08 | 8.388e-08 | 1.9e-06 | 1.778e-06 | 3.2e-03 | 3.169e-03 | 1, 1 |
| 2048 | 2.1e-08 | 2.097e-08 | 6.8e-07 | 6.287e-07 | 2.2e-03 | 2.243e-03 | 1, 1 |
| 4096 | 5.1e-09 | 5.242e-09 | 2.4e-07 | 2.223e-07 | 1.6e-03 | 1.587e-03 | 1, 1 |
| 8192 | 1.3e-09 | 1.305e-09 | 8.5e-08 | 7.858e-08 | 1.1e-03 | 1.123e-03 | 1, 1 |

Ratios 1.00 to 1.05 (T1), 0.92 to 0.95 (T3), 0.99 to 1.03 (T2); orders: published 1.99, 1.50, 0.50, ours 2.00, 1.50, 0.50.

Level: order of magnitude for both tables.
18 of the 42 values agree with the published ones to the printed two digits (T2 red-black from `N = 256` on, T2 lexicographic from 1024 on, T1 at four `N` red-black and three lexicographic, T3 red-black at 256), the others differ in the last digit, so the level is not exact.
The paper's observation holds in ours: red-black is the more accurate ordering on T1 and T3, by 20 to 21 percent (T1) and 11 percent (T3) against the paper's 13 to 30 and 12 to 15 percent.
After one cycle ours is 6 percent above the discrete solution's error with red-black sweeps (T1 and T3), 28 percent (T1) and 17 percent (T3) above it with lexicographic ones, and within 2 percent on T2: one FMG cycle reaches the discretisation error, as the paper says, with an algebraic error smaller than it.

## Table 3: T9

| N | Published | Ours | As printed | Ratio | Red-black (ours) | Converged | Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| 8 | 1.2e-03 | 1.507e-03 | **1.5e-03** | 1.26 | 1.396e-03 | 1.157e-03 | 1, 1 |
| 16 | 2.7e-04 | 4.107e-04 | **4.1e-04** | 1.52 | 3.786e-04 | 2.977e-04 | 1, 1 |
| 32 | 5.7e-05 | 1.083e-04 | **1.1e-04** | 1.90 | 9.597e-05 | 7.551e-05 | 1, 1 |
| 64 | 1.4e-05 | 2.786e-05 | **2.8e-05** | 1.99 | 2.384e-05 | 1.891e-05 | 1, 1 |
| 128 | 6.2e-06 | 7.065e-06 | **7.1e-06** | 1.14 | 5.905e-06 | 4.730e-06 | 1, 1 |

Orders: published 1.95, ours 1.94.
Level: order of magnitude, with the ratio at `N = 64` at the edge of `[1/2, 2]` (Finding 1).
The paper's order drops to 1.2 from `N = 64` to 128, which it puts down to rounding (the operator divides by `h^6`); the discrete solution's error falls by a factor 4.0 there (1.891e-5 to 4.730e-6), so in our float64 runs the second order holds at 128.
The Gauss-Seidel column:

| N | Published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| 8 | 1.2e-03 | 1.158e-03 | 72, 72 |
| 16 | 3.0e-04 | 2.984e-04 | 270, 270 |
| 32 | 8.1e-05 | 8.137e-05 | 867, 867 |

Level: exact, with the same number of sweeps.

## Table 5: T46

| N | Published | Ours | As printed | Ratio | Red-black (ours) | Converged | Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| 8 | 1.7e-02 | 1.854e-02 | **1.9e-02** | 1.09 | 1.523e-02 | 1.712e-02 | 2, 1 |
| 16 | 3.3e-03 | 4.999e-03 | **5.0e-03** | 1.51 | 4.104e-03 | 4.304e-03 | 1, 1 |
| 32 | 8.1e-04 | 1.296e-03 | **1.3e-03** | 1.60 | 1.042e-03 | 1.078e-03 | 1, 1 |
| 64 | 2.1e-04 | 3.276e-04 | **3.3e-04** | 1.56 | 2.621e-04 | 2.695e-04 | 1, 1 |
| 128 | 5.2e-05 | 8.218e-05 | **8.2e-05** | 1.58 | 6.568e-05 | 6.738e-05 | 1, 1 |

Orders: published 2.07, ours 1.96.
Level: order of magnitude (Finding 1).

| N | Gauss-Seidel published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| 8 | 1.7e-02 | 1.712e-02 | 82, 82 |
| 16 | 4.3e-03 | 4.304e-03 | 296, 296 |
| 32 | 1.1e-03 | 1.073e-03 | 954, 954 |

Level: exact, with the same number of sweeps.

## Table 6: T28

| N | Published | Ours | As printed | Ratio | Red-black (ours) | Converged |
|---|---|---|---|---|---|---|
| 8 | 5.3e-04 | 4.386e-04 | **4.4e-04** | 0.83 | 4.160e-04 | 4.040e-04 |
| 16 | 2.1e-04 | 1.673e-04 | **1.7e-04** | 0.80 | 1.555e-04 | 1.438e-04 |
| 32 | 7.4e-05 | 6.087e-05 | **6.1e-05** | 0.82 | 5.553e-05 | 5.090e-05 |
| 64 | 2.6e-05 | 2.171e-05 | **2.2e-05** | 0.84 | 1.967e-05 | 1.800e-05 |
| 128 | 9.3e-06 | 7.691e-06 | **7.7e-06** | 0.83 | 6.957e-06 | 6.364e-06 |

Orders: published 1.47, ours 1.46; one cycle at every `N` in both.
Level: order of magnitude.
Here the published errors are above ours, by a constant 20 percent, and 31 to 46 percent above the discrete solution's, unlike Tables 3 and 5 (Finding 1).

| N | Gauss-Seidel published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| 8 | 4.0e-04 | 4.040e-04 | 86, 86 |
| 16 | 1.4e-04 | 1.438e-04 | 312, 312 |
| 32 | 5.1e-05 | 5.091e-05 | 925, 924 |

Level: exact.

## Table 7: T22

| N | Published | Ours | As printed | Ratio | Red-black (ours) | Converged |
|---|---|---|---|---|---|---|
| 8 | 1.4e-03 | 1.392e-03 | 1.4e-03 | 0.99 | 1.388e-03 | 1.417e-03 |
| 16 | 1.4e-03 | 1.359e-03 | 1.4e-03 | 0.97 | 1.358e-03 | 1.377e-03 |
| 32 | 4.3e-03 | 1.087e-03 | **1.1e-03** | 0.25 | 1.084e-03 | 1.102e-03 |
| 64 | 2.2e-03 | 2.357e-03 | **2.4e-03** | 1.07 | 2.628e-03 | 8.367e-04 |
| 128 | 1.6e-03 | 1.987e-03 | **2.0e-03** | 1.24 | 4.059e-03 | (6.372e-04) |

Orders: published -0.10, ours -0.18; one cycle at every `N` in both.
Level: mismatch at `N = 32` (Finding 4); at the other four `N` the ratio lies in `[0.97, 1.24]`.

| N | Gauss-Seidel published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| 8 | 1.4e-03 | 1.417e-03 | 75, 75 |
| 16 | 1.4e-03 | 1.377e-03 | 266, 266 |
| 32 | 1.1e-03 | 1.102e-03 | 854, 851 |

Level: exact.
At `N = 128` the cycles do not settle on a discrete solution (Finding 4); the "converged" value in parentheses is the tenth iterate of a periodic sequence.

## Finding 1: on the smooth 3D examples one published FMG cycle lands below the discrete solution's error

On T9 and T46 the published FMG errors lie below the error of the discrete solution, by 9 to 26 percent on T9 from `N = 16` to 64 (2.7e-4 against 2.98e-4, 5.7e-5 against 7.55e-5, 1.4e-5 against 1.89e-5) and by 22 to 25 percent on T46 at every `N` from 16 (3.3e-3 against 4.30e-3, 8.1e-4 against 1.08e-3, 2.1e-4 against 2.70e-4, 5.2e-5 against 6.74e-5); at `N = 128` on T9 the published value is above it (6.2e-6 against 4.73e-6), where the paper notes round-off.
The discrete solution is the one the paper solves: its own Gauss-Seidel column, which stops near it, agrees with ours to the printed digits and to the sweep at every `N`, a sweep-by-sweep test of the scheme, the root selection and the lexicographic order.
So the published FMG iterate carries an algebraic error that cancels part of the discretisation error, while ours, after the same one cycle with the lexicographic order, lies 30 to 49 percent (T9) and 8 to 22 percent (T46) above the discrete solution's error.
Our red-black cycle shows that the sign of the algebraic error depends on the details: on T46 it lands 3 to 11 percent below the discrete solution's error, on T9 21 to 27 percent above.
What differs is not in the paper: the "cubic interpolation" of the FMG step (ours: the not-a-knot cubic spline, `docs/methods/multigrid.md`), the order of the loops within a lexicographic sweep (on these problems, symmetric in the axes, a permutation of the axes cannot matter), or the residual the paper measures.
On T28 the published errors lie 31 to 46 percent above the discrete solution's and ours 9 to 21 percent above it; in 2D, where the same paper's FMG errors also lie above the discrete solution's (T1: 4.5e-6 against 4.20e-6 at `N = 128`), ours agree with the published ones to within 8 percent.
The levels stand, but on T9 at `N = 64` only just (ratio 1.99).

## Finding 2: the reference of the relative residual

The paper stops at `||r_l||_2 / ||r_0||_2 <= 1e-6` and does not say what `r_0` is for the FMG-FAS method, which has no initial guess on the finest grid.
Ours is the residual of the node values that are 0 inside, with the Dirichlet data on the ring (the iterate from which the coarsest grid starts), a residual that grows like `N^5` in 3D at the nodes next to the ring.
With it the iteration counts on T2 match Table 1 in all seven cells and Table 2 in five (ours 2 against 3 at `N = 128`, 2 against 1 at 512), and every other example takes one cycle, as published, except T46 at `N = 8` (ours 1, published 2).
The relative residuals after the cycle do not match: in 3D ours are 1.2 to 230 times smaller than the printed ones, the gap growing with `N`, so the paper's reference grows more slowly with `N` than ours.
For the stand-alone Gauss-Seidel solver the natural reference, the residual of the start `1.01 u`, reproduces the published sweep counts exactly, so the norm itself (the plain L2 norm over the unknowns) is the paper's.
The stopping rule decides only whether a second cycle runs; where it does (T2 at `N = 128` and 256, and 512 lexicographic), the error changes by 0.4 to 1.7 percent.

## Finding 3: the smallest real root also where the coarse right-hand side is negative

The theorems of Section 2 select the root with a positive definite Hessian for `f > 0`; the right-hand side of a coarse grid in a FAS cycle, `S_H(I w) + I r`, can be negative next to a singularity, where `r` is large.
The paper's rule, "the smallest of all the real roots" (eq. (12)), is what `smallest_root` does there; it leaves that coarse node with an indefinite Hessian, which only the coarse correction sees.
The alternative of clamping `b` at 0 (the positive semidefinite limit) stalls the cycles on T2 (`multigrid_checks.csv`): the relative residual stops at 1.9e-5 (`N = 128`) to 1.7e-6 (`N = 1024`) and none of the 8 clamped runs reaches `1e-6` in 50 cycles, with errors of 7.1e-3 to 1.1e-2 against the paper's 3.2e-3 to 8.9e-3, while the smallest real root converges in 1 or 2 cycles to the published errors.

## Finding 4: T22 at `N = 32`, and the cycles next to the corner

The published FMG error on T22 jumps from 1.4e-3 at `N = 16` to 4.3e-3 at 32 and falls again (2.2e-3, 1.6e-3); the paper's own Gauss-Seidel column, and ours, give 1.1e-3 at 32, and our FMG cycle gives 1.09e-3, within 1.4 percent of the discrete solution's error, so the published value is 4 times all three.
So `N = 32` is a mismatch; the paper prints nothing that would explain it.

At `N = 64` and 128 our FMG cycle stops above the discrete solution's error too (2.36e-3 against 8.37e-4 at 64), with a residual of order `10^3` to `10^4` at the nodes next to the corner `(1, 1, 1)`, where `f` grows like `h^-5`, and a few nodes with an indefinite discrete Hessian there (non-convex fraction 7.2e-5 at 64, 1.1e-5 at 128); the relative residual meets `1e-6` because its reference is dominated by the ring.
`multigrid_checks.csv` runs more cycles: at `N = 64` they converge (the error 8.3671e-4 from cycle 10 on, the residual's max 1e-9 from cycle 20), at `N = 128` they do not.
There the iterates repeat: with red-black sweeps cycle 20 equals cycle 5 and cycle 30 cycle 15, with lexicographic ones cycle 30 equals cycle 10, to all digits, and the error moves between 6.4e-4 and 2.0e-3 while the residual's max stays between 3e2 and 1e5.
So at `N = 128` the multigrid iteration has no limit near the corner, and the published 1.6e-3 and our 1.99e-3 are both points on such a path; the "converged" value of Table 7 at 128 is the tenth red-black cycle.
Whether a smaller stopping step near the corner, more sweeps there, or another coarse-grid treatment would converge is not tested here.

T2, the 2D version, converges (10 cycles bring the relative residual to `1e-11`), but at `N >= 512` the cycle that meets the tolerance leaves a residual of up to `3e8` at the corner node (`f` there is about `1/(8 h^2)`) and non-convex fractions of 2e-6 to 5e-5, while the error is within 2 percent of the discrete solution's.

## Conventions

- Grid: `N` cells per side, `h = 1/N` (`pi / N` on T46); the paper defines its grid this way (Section 2), so ADR 0001 is the paper's own statement here.
- The paper does not say which cubic interpolation the FMG step uses, nor what `r_0` is (Findings 1, 2).
- The red-black order updates each colour at once; with the diagonal neighbours of a 9- or 19-point stencil in the same colour this is a Jacobi step within the colour, as in the paper's vectorised MATLAB (Section 4.1).
- In 3D the paper sweeps lexicographically only; our red-black columns are ours.

## Iterations and time

The time is `Result.info["time"]` (the discretisation of every grid, the cycles and the solution fields, the spline through the node values among them) on one thread of an Intel Xeon Platinum 8358 of a100server, six runs at a time on a shared machine; the paper's is MATLAB on an Intel Core i3-3120M laptop.
In 2D the red-black time grows by a factor of 4.6 to 5.6 per halving of `h` from `N = 1024` on, about the factor 4 of the unknowns, as the paper's does; the lexicographic one by 2.6 to 4.8, since the fixed cost of each of its `3 N` vectorised steps dominates below `N = 8192`.
At `N = 8192` ours takes 154 s (red-black) and 125 s (lexicographic) against the paper's 67 s and 246 s.
In 3D ours grows by 2 to 4 per halving up to `N = 128`, less than the factor 8 of the unknowns, for the same reason; at `N = 128` it takes 8 to 11 s against the paper's 133 to 137 s.
The times are recorded, not compared (ADR 0002).

## Not compared

- Table 4: the wide-stencil hybrid finite differences of Froese-Oberman (J. Comput. Phys. 2011) and the quadratic and cubic `C^0` interior-penalty elements of Brenner-Neilan (2012) on T9, reprinted from those papers.
  The 3D monotone scheme (#27) and the interior-penalty method (0.3) will compare with them; neither is in maesolvers yet.
- The Gauss-Seidel CPU times, and the Gauss-Seidel columns beyond `N = 32`, which the paper does not print.
