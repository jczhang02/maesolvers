# Standard 9-point finite differences: published versus ours

Problems: P4 (smooth, `u = exp(|x|^2 / 2)`) and P5 (corner blow-up, `u = -sqrt(2 - |x|^2)`) on `(0, 1)^2`, Dirichlet data from the exact solution.
Published numbers: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Standard"; the same values appear in Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 3, column "Standard F.D.".
Both papers credit the column to the Gauss-Seidel and Poisson iterations of Benamou-Froese-Oberman (2010).
Full transcription and caveats: `sources.md`.

Ours: `FDPoisson` (the 2010 Poisson iteration, successive-iterate tolerance 1e-12) and `FDNewton` (damped Newton from the Poisson start, residual tolerance 1e-10 relative to `max f`), both on the same discretisation, float64, scipy sparse LU.
`FDPoisson` is the preset `PoissonIteration(scheme=Standard9(n=N))` and `FDNewton` is `Newton(scheme=Standard9(n=N))` (ADR 0004).
Raw data: `fd_poisson_sequence.csv`, `fd_newton_sequence.csv`, written by `run_standard_fd.py` (`just remote python benchmarks/run_standard_fd.py`, which also logs every solve to MLflow under `maesolvers-fdpoisson-P4` and the like).
Error: max over all grid nodes of `|u_h - u|`.

## Finding 1: the grid convention is `h = 1 / N`

The 2011 papers say only "an N x N grid".
Three conventions were tried on P4 at N = 31 against the published 7.14e-5:

| Convention | Cells per side | Our max error |
|---|---|---|
| `h = 1 / (N - 1)` | 30 | 7.617e-5 |
| `h = 1 / N` | 31 | 7.144e-5 |
| `h = 1 / (N + 1)` | 32 | 6.696e-5 |

`h = 1 / N` matches to four digits, and keeps matching at every N below.
So the papers' N is the number of cells per side (the 2010 paper's "h = L / N"), and the grid has `(N + 1)^2` nodes.

## P4, smooth example

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

## P5, example with blow-up

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

## Finding 2: Newton on the standard scheme picks a non-convex root on P5

`FDNewton` converges in the residual at every N (7 to 23 iterations), but to a different root of the discrete equations: non-convex at 0.1 to 1.3 percent of the interior nodes, all near the corner `(1, 1)`, with a smallest Hessian eigenvalue that grows without bound as N grows.
Its max error stays near 2e-2 and does not decrease with N.
Froese-Oberman (JCP 2011, Section 8.1) report that Newton on the standard scheme fails on this example even from the exact solution, oscillating into a non-convex iterate; our run from the Poisson start lands on such a non-convex discrete solution and stays there.
The discrete equation `u_xx u_yy - u_xy^2 = f` does not see the sign of the Hessian, so nothing in the Newton iteration prefers the convex root; the Poisson iteration's right-hand side is the Laplacian of a convex solution, which is why it selects the convex branch.

Consequence for the library: the standard scheme is two methods, not one.
Its fixed point (Poisson or Gauss-Seidel iteration) is what the literature's "standard F.D." column means; Newton on the same discretisation needs a convexity-selecting safeguard before it can be reported on singular problems.

## Finding 3: on P5, rounding decides which non-convex root Newton reaches

The receipts were regenerated on a100server with `run_standard_fd.py` after the scheme and the iteration became separate parts (ADR 0004).
Every `FDPoisson` row and every `FDNewton` row on P4 agrees with the first receipts to about 3e-9 relative or better (`residual_max`, a rounding-level number, to a few percent).
`FDNewton` on P5 moved at N = 181 with `h = 1/(N - 1)`, at N = 255 and at N = 361: at N = 361 the max error went from 21.94e-3 to 22.25e-3 and the iterations from 18 to 23.
The code before the change (commit 97fad98) gives the same numbers on a100server as the code after it, bit for bit in every column but the time, so the code is not the cause: the first receipts came from another machine and library build.
This is Finding 2 from another side: nothing in Newton's iteration selects a root, so near the corner rounding at the level of the last bit decides which non-convex discrete solution it lands on.
The published comparison uses `FDPoisson`, which is unaffected.

## Not compared

CPU times (2010 MATLAB hardware), the monotone and hybrid columns (schemes not implemented yet), and the Gauss-Seidel iteration counts (method not implemented).

## Shared 201 x 201 grid

The protocol asks for the error on the shared grid as well (cubic tensor spline off the nodes).
P4 agrees with the node error, as it should for a smooth solution.
P5 is larger on the shared grid at coarse N: the spline overshoots near the corner where the gradient blows up, so for P5 the node error is the number to compare with the literature and the shared-grid error measures the interpolant as much as the scheme.

| N | P4 nodes | P4 shared grid | P5 nodes | P5 shared grid |
|---|---|---|---|---|
| 31 | 7.144e-05 | 7.144e-05 | 1.715e-02 | 4.474e-02 |
| 63 | 1.731e-05 | 1.732e-05 | 1.253e-02 | 2.609e-02 |
| 127 | 4.263e-06 | 4.263e-06 | 9.004e-03 | 8.023e-03 |
| 255 | 1.057e-06 | 1.057e-06 | 6.418e-03 | 5.645e-03 |
