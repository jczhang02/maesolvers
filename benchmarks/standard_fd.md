# Standard 9-point finite differences: published versus ours

Problems: T1 (smooth, `u = exp(|x|^2 / 2)`) and T2 (corner blow-up, `u = -sqrt(2 - |x|^2)`) on `(0, 1)^2`, Dirichlet data from the exact solution.
Published numbers: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Standard"; the same values appear in Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 3, column "Standard F.D.".
Both papers credit the column to the Gauss-Seidel and Poisson iterations of Benamou-Froese-Oberman (2010).
Problem T7 (`f = 1`, `g = 1` on `[-1, 1]^2`, no exact solution) is compared with Benamou-Froese-Oberman, ESAIM: M2AN 44 (2010), Table 7, column "(M1), (M2)", in its own section below.
Full transcription and caveats: `sources.md`.

Ours: `FDPoisson` (the 2010 Poisson iteration, successive-iterate tolerance 1e-12) and `FDNewton` (damped Newton from the Poisson start, residual tolerance 1e-10 relative to `max f`), both on the same discretisation, float64, scipy sparse LU.
`FDPoisson` is the preset `PoissonIteration(scheme=Standard9(n=N))` and `FDNewton` is `Newton(scheme=Standard9(n=N))` (ADR 0004).
Raw data: `fd_poisson_sequence.csv`, `fd_newton_sequence.csv`, written by `run_standard_fd.py` (`just remote python benchmarks/run_standard_fd.py`, which also logs every solve to MLflow under `maesolvers-fdpoisson-T1` and the like); for T7, `t7_bfo2010.csv`, written by `t7_bfo2010.py` on a100server.
Error: max over all grid nodes of `|u_h - u|`.

## Finding 1: the grid convention is `h = 1 / N`

The 2011 papers say only "an N x N grid".
Three conventions were tried on T1 at N = 31 against the published 7.14e-5:

| Convention | Cells per side | Our max error |
|---|---|---|
| `h = 1 / (N - 1)` | 30 | 7.617e-5 |
| `h = 1 / N` | 31 | 7.144e-5 |
| `h = 1 / (N + 1)` | 32 | 6.696e-5 |

`h = 1 / N` matches to four digits, and keeps matching at every N below.
So the papers' N is the number of cells per side (the 2010 paper's "h = L / N"), and the grid has `(N + 1)^2` nodes.

## T1, smooth example

| N | Published | FDPoisson | FDNewton | Poisson iterations | Newton iterations |
|---|---|---|---|---|---|
| 31 | 7.14e-5 | 7.144e-5 | 7.144e-5 | 49 | 4 |
| 45 | 3.39e-5 | 3.391e-5 | 3.391e-5 | 49 | 4 |
| 63 | 1.73e-5 | 1.731e-5 | 1.731e-5 | 50 | 4 |
| 89 | 0.87e-5 | 8.678e-6 | 8.678e-6 | 50 | 4 |
| 127 | 0.43e-5 | 4.263e-6 | 4.263e-6 | 50 | 4 |
| 181 | 0.21e-5 | | 2.099e-6 | | 4 |
| 255 | 0.11e-5 | | 1.057e-6 | | 4 |
| 361 | 0.05e-5 | | 5.276e-7 | | 4 |

Reproduced: every value agrees with the published one to the digits printed, order `h^2`.
Both solvers reach the same discrete solution, as the 2010 paper states for its two iterations.
Newton needs 4 iterations at every N; the papers never tabulate Newton on the standard scheme, so there is nothing to compare that against.

Level (ADR 0002), per solver:

- `FDPoisson`: exact at N = 31 to 127, the N its receipt covers (`POISSON_T1_UP_TO` in `run_standard_fd.py`); it was not run at N = 181 to 361.
- `FDNewton`: exact at every printed N, against a column the papers credit to the Poisson and Gauss-Seidel iterations, not to Newton.
  Newton reaches the same discrete solution as those iterations, so its values match the column; this is a check of the discrete solution, not a reproduction of a published Newton run.

`tests/test_reproduction.py` pins N = 31 and 63 for both solvers.

## T2, example with blow-up

| N | Published | FDPoisson | FDNewton | FDNewton min Hessian eigenvalue |
|---|---|---|---|---|
| 31 | 17.15e-3 | 17.15e-3 | 17.45e-3 | -11.7 |
| 45 | 14.59e-3 | 14.59e-3 | 15.96e-3 | -50.1 |
| 63 | 12.53e-3 | 12.53e-3 | 19.31e-3 | -94.0 |
| 89 | 10.67e-3 | 10.67e-3 | 16.52e-3 | -149.0 |
| 127 | 9.00e-3 | 9.004e-3 | 18.79e-3 | -255.3 |
| 181 | 7.59e-3 | 7.587e-3 | 21.63e-3 | -1463.9 |
| 255 | 6.42e-3 | 6.418e-3 | 19.70e-3 | -807.1 |
| 361 | 5.41e-3 | 5.409e-3 | 22.25e-3 | -1182.3 |

Reproduced with the Poisson iteration: every value agrees to the digits printed, order about `h^0.5`, and the discrete solution is convex at every node (smallest Hessian eigenvalue 0.71 at every N).

Level (ADR 0002), per solver:

- `FDPoisson`: exact at every printed N.
  `tests/test_reproduction.py` pins N = 31 and 63.
- `FDNewton`: mismatch.
  It converges to a non-convex discrete solution at every N, whose error does not decrease with N (Finding 2).

## Finding 2: Newton on the standard scheme picks a non-convex root on T2

`FDNewton` converges in the residual at every N (7 to 23 iterations), but to a different root of the discrete equations: non-convex at 0.1 to 1.3 percent of the interior nodes, all near the corner `(1, 1)`, with a smallest Hessian eigenvalue that grows without bound as N grows.
Its max error stays near 2e-2 and does not decrease with N.
Froese-Oberman (JCP 2011, Section 8.1) report that Newton on the standard scheme fails on this example even from the exact solution, oscillating into a non-convex iterate; our run from the Poisson start lands on such a non-convex discrete solution and stays there.
The discrete equation `u_xx u_yy - u_xy^2 = f` does not see the sign of the Hessian, so nothing in the Newton iteration prefers the convex root; the Poisson iteration's right-hand side is the Laplacian of a convex solution, which is why it selects the convex branch.

Consequence for the library: the standard scheme is two methods, not one.
Its fixed point (Poisson or Gauss-Seidel iteration) is what the literature's "standard F.D." column means; Newton on the same discretisation needs a convexity-selecting safeguard before it can be reported on singular problems.

## T7, BFO 2010 Table 7

The table prints `u_min`, the minimum of the discrete solution, at N = 21 to 141.
Ours: `u_origin`, the value at the origin, at `n = N - 1` cells per side, where the origin is a node (Finding 3); it equals the nodal minimum in every run below.
`FDPoisson` runs with `max_iter = 100000`, since the Poisson iteration needs about `n^2` iterations here (14978 at n = 140); `FDNewton` with its defaults.

| N | n | Published | FDPoisson | FDNewton | FDNewton non-convex nodes | FDNewton min Hessian eigenvalue | Poisson iterations |
|---|---|---|---|---|---|---|---|
| 21 | 20 | 0.2892 | 0.2892 | 0.2892 | 0 | 0.10 | 438 |
| 41 | 40 | 0.2734 | 0.2734 | 0.3119 | 5.3% | -3.7 | 1539 |
| 61 | 60 | 0.2682 | 0.2682 | 0.3189 | 4.4% | -10.4 | 3225 |
| 81 | 80 | 0.2655 | 0.2655 | 0.3134 | 4.6% | -11.3 | 5443 |
| 101 | 100 | 0.2639 | 0.2639 | 0.3249 | 6.5% | -15.1 | 8159 |
| 121 | 120 | 0.2629 | 0.2629 | 0.3206 | 5.6% | -12.8 | 11344 |
| 141 | 140 | 0.2621 | 0.2621 | 0.3245 | 6.8% | -19.1 | 14978 |

Level (ADR 0002), per solver:

- `FDPoisson`: exact.
  Every value rounds to the printed one at four decimals, and the discrete solution is convex at every node (smallest Hessian eigenvalue 0.015 at n = 140).
  `tests/test_reproduction.py` pins N = 21, 41 and 61.
- `FDNewton`: mismatch.
  It agrees at N = 21 only; from N = 41 on it converges in the residual (8 to 15 iterations) to a non-convex discrete solution, non-convex at 4 to 7 percent of the nodes, whose value at the origin stays near 0.31 to 0.32 and does not approach the published values.
  This is the behaviour of Finding 2 on T2, now on a problem whose singularity sits at the four corners.
  `tests/test_fd_newton.py` keeps N = 21 and N = 41 as a regression check, not as a reproduction.

## Finding 3: Table 7 counts nodes, not cells

On `[-1, 1]^2` the 2010 paper's own rule `h = L / N` would give `n = N` cells, the convention that matched T1 and T2 (Finding 1).
Table 7 does not follow it:

| Convention | Cells per side at N = 21 | Origin a node | Our `u_origin` | Our nodal minimum |
|---|---|---|---|---|
| `h = 2 / (N - 1)` | 20 | yes | 0.2892 | 0.2892 |
| `h = 2 / N` | 21 | no | 0.2877 | 0.2900 |

Published: 0.2892.
`h = 2 / (N - 1)` matches at every printed N above; `h = 2 / N` matches neither the value at the origin nor the nodal minimum.
So in Table 7, N is the number of nodes per side, the boundary included, and the grid has `N^2` nodes.
So Table 7 does not follow the rule the same paper states in Section 3.3; whether its other tables follow it was not checked.

## Finding 4: on T2, rounding decides which non-convex root Newton reaches

The receipts were regenerated on a100server with `run_standard_fd.py` after the scheme and the iteration became separate parts (ADR 0004).
Every `FDPoisson` row and every `FDNewton` row on T1 agrees with the first receipts to about 3e-9 relative or better (`residual_max`, a rounding-level number, to a few percent).
`FDNewton` on T2 moved at N = 181 with `h = 1/(N - 1)`, at N = 255 and at N = 361: at N = 361 the max error went from 21.94e-3 to 22.25e-3 and the iterations from 18 to 23.
The code before the change (commit 97fad98) gives the same numbers on a100server as the code after it, bit for bit in every column but the time, so the code is not the cause: the first receipts came from another machine and library build.
This is Finding 2 from another side: nothing in Newton's iteration selects a root, so near the corner rounding at the level of the last bit decides which non-convex discrete solution it lands on.
The published comparison uses `FDPoisson`, which is unaffected.

## Not compared

CPU times (2010 MATLAB hardware) and the Gauss-Seidel iteration counts (method not implemented); the monotone columns are compared in `monotone17.md`, the hybrid column in `hybrid.md`.
For T7: the 9-, 17- and 33-point wide-stencil columns of Table 7 (scheme not implemented yet) and the iteration counts and times of Table 8 (stopping rule `10^-14` on successive iterates against our `10^-12`, and 2010 hardware).

## Shared 201 x 201 grid

The protocol asks for the error on the shared grid as well (cubic tensor spline off the nodes).
T1 agrees with the node error, as it should for a smooth solution.
T2 is larger on the shared grid at coarse N: the spline overshoots near the corner where the gradient blows up, so for T2 the node error is the number to compare with the literature and the shared-grid error measures the interpolant as much as the scheme.

| N | T1 nodes | T1 shared grid | T2 nodes | T2 shared grid |
|---|---|---|---|---|
| 31 | 7.144e-05 | 7.144e-05 | 1.715e-02 | 4.474e-02 |
| 63 | 1.731e-05 | 1.732e-05 | 1.253e-02 | 2.609e-02 |
| 127 | 4.263e-06 | 4.263e-06 | 9.004e-03 | 8.023e-03 |
| 255 | 1.057e-06 | 1.057e-06 | 6.418e-03 | 5.645e-03 |
