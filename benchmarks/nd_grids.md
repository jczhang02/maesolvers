# n-dimensional grids: the cost of the spline on the shared evaluation grid

Issue #25 makes the grid of the finite-difference schemes (`BoxGrid`), its boundary ring, the Dirichlet data, the centred node Hessian and the tensor spline that `evaluate` reads on the shared evaluation grid work in any dimension.
No paper is reproduced here, so none of the five levels of ADR 0002 applies: this file records what a 3D or 4D grid method costs `evaluate`, and the interpolation error it reports, so that the 19-point scheme (#27) and multigrid (#30) start from known numbers.
`tests/test_grid.py` checks the same construction at small `n`; nothing is pinned.

Raw data: `nd_grids.csv`, written by `run_nd_grids.py` on a100server at commit e89640c (`just remote-bg nd-grids env OMP_NUM_THREADS=16 python benchmarks/run_nd_grids.py`; 16 threads, the machine shared with other runs), which also logs every row to MLflow under `maesolvers-nd-grids-T9` and `maesolvers-nd-grids-T10`.

## The setting

T9 is `u = exp(|x|^2/2)` on `(0, 1)^3`, T10 the same on `(0, 1)^4`; `n` cells per side, `h = 1/n` (ADR 0001).
Each row builds `BoxGrid(problem.domain, n)`, takes the exact solution at its `(n + 1)^d` nodes, fits the tensor spline of degree 1, 3 (the schemes' default `spline_degree`) or 5 through them, and builds the `Result` a grid method would return, with `centred_hessian` at the interior nodes.
The shared evaluation grid has 51^3 = 132651 points in 3D and 21^4 = 194481 in 4D (ADR 0007, provisional).
Times are wall-clock seconds: `fit_s` to fit the spline, `eval_s` to evaluate it on the shared grid (the best of three), `evaluate_s` for the whole of `evaluate`.
`peak_rss_mb` is the peak resident memory of the process so far, so a row inherits the peak of the rows before it.

In 3D and 4D the spline is the tensor product of scipy's `make_interp_spline` (not-a-knot), fitted one axis after the other and evaluated by `NdBSpline`; in 2D it stays FITPACK's `RectBivariateSpline`, so that 2D results do not change.
On the box the two agree to about `1e-15` times the largest value, which `tests/test_grid.py` checks on 2D axes; off the box they differ (FITPACK clamps the points to the box, `NdBSpline` extends the end polynomials), but `evaluate` reads the spline only on the box.

## Results

Every row has `max = 0` and `nonconvex_fraction = 0`: with exact values at the nodes `evaluate` sees no error there, and the centred Hessian of the convex `exp(|x|^2/2)` is positive definite at every interior node, in 3D and in 4D.
`max_grid` is the spline's error on the shared grid; the order is `log2` of the ratio to the same degree at the previous `n`.

T9 (3D), 51^3 shared points:

| n | nodes | degree | `fit_s` | `eval_s` | `evaluate_s` | `max_grid` | order |
|---|---|---|---|---|---|---|---|
| 8 | 729 | 3 | 0.008 | 0.059 | 0.074 | 6.14e-4 | |
| 16 | 4913 | 3 | 0.005 | 0.059 | 0.13 | 4.68e-5 | 3.7 |
| 32 | 35937 | 3 | 0.006 | 0.060 | 0.15 | 2.24e-6 | 4.4 |
| 64 | 274625 | 3 | 0.039 | 0.063 | 0.74 | 6.92e-8 | 5.0 |
| 128 | 2146689 | 3 | 0.16 | 0.070 | 5.3 | 7.60e-10 | 6.5 |
| 128 | 2146689 | 1 | 0.033 | 0.026 | 8.7 | 1.87e-4 | 1.9 |
| 128 | 2146689 | 5 | 0.17 | 0.18 | 5.7 | 2.67e-13 | 7.2 |

T10 (4D), 21^4 shared points:

| n | nodes | degree | `fit_s` | `eval_s` | `evaluate_s` | `max_grid` | order |
|---|---|---|---|---|---|---|---|
| 4 | 625 | 3 | 0.003 | 0.31 | 0.34 | 1.26e-2 | |
| 8 | 6561 | 3 | 0.005 | 0.31 | 0.34 | 1.26e-3 | 3.3 |
| 16 | 83521 | 3 | 0.021 | 0.32 | 0.54 | 3.54e-5 | 5.2 |
| 32 | 1185921 | 3 | 0.16 | 0.32 | 4.8 | 1.98e-6 | 4.2 |
| 32 | 1185921 | 1 | 0.018 | 0.038 | 4.6 | 5.45e-3 | 1.7 |
| 32 | 1185921 | 5 | 0.15 | 1.5 | 5.6 | 4.04e-9 | 6.4 |

The CSV has every degree at every `n`.

## What the numbers say

1. The spline onto the shared grid is cheap.
   Fitting it takes at most 0.17 s, at two million nodes; evaluating it on the shared grid takes 0.06 to 0.07 s in 3D and 0.32 s in 4D at the default degree 3, whatever `n`, since it costs `(k + 1)^d` basis products per point: 64 in 3D, 256 in 4D.
   Degree 5 (1296 products per point in 4D) takes 1.5 s in 4D, degree 1 under 0.04 s.
   So `max_grid` adds well under a second to a 3D or 4D solve, and the shared grid sizes of ADR 0007 stay practical.
2. At large `n` the cost of `evaluate` is at the nodes, not on the shared grid: 5 to 9 s at 2.1 million nodes in 3D and about 5 s at 1.2 million in 4D, for the smallest eigenvalue of the node Hessian at every interior node and the exact solution's Hessian by finite differences at the same nodes (`eigmin_exact`).
   It grows with the number of nodes, as the solve itself does, and is small beside a solve on such a grid.
   Building the nodes takes 0.24 s at `n = 128` in 3D; the process peaks at 1.4 GB there.
3. The spline converges at least as fast as its degree allows: about order 2 for degree 1, 4 to 6.5 for degree 3 from `n = 16` on, and 6 to 7 for degree 5.
   Orders above `k + 1` come from taking the max over fixed points of the shared grid, where the error of the spline is not at its peak; they are not a property to rely on.
   At the default degree the interpolation error is far below the `O(h^2)` error of a second-order scheme at the nodes, so in 3D and 4D, as in 2D, `max_grid` tracks the scheme's error, not the interpolant's.

## The 2D path is unchanged

`Standard9`, `Monotone17` and, through them, `Hybrid` and `Filtered` now build their grids with `BoxGrid`, and `difference_operators`, which `Standard9` and `Monotone17` call, is now the 2D case of `second_differences`.
On a100server, every preset (`FDNewton`, `FDPoisson`, `FDMonotone` with 9, 17 and 33 points and both boundary treatments, `FDHybrid`, `FDFiltered`) on T1, T2, T20 and, for the standard scheme, T8 and T6, at `n = 8`, 16 and 25 and spline degrees 1 to 5, gives node values, node Hessians, the spline on the shared grid, `evaluate`'s fields and the final residual that are equal, as float arrays with `numpy.array_equal`, to those of `next` at 4fdd5da (912 arrays); so do the matrices of `difference_operators` and of `directional_difference`.
So no 2D receipt in `benchmarks/` changes.
