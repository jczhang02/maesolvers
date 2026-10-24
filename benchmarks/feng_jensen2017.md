# Semi-Lagrangian scheme: published versus ours

`FDSemiLagrangian` (`Howard` on `SemiLagrangian`) against Feng and Jensen, SIAM J. Numer. Anal. 55 (2017), Section 8, read from the journal PDF (`sources.md`, section 7).
**Levels (ADR 0002): figure only on T39**, against Figure 6, a table of the best relative error over the stencil factor `m` at each level under a figure caption, and the rate `h^1.02` of p. 711; **mismatch on T40**, which our structured grid solves to the rounding level (Finding 1).
The paper's nonsmooth behaviour is compared on T40 turned off the grid, a test-local problem without a T id (Finding 2).

Receipt: `feng_jensen2017.csv`, written by `run_feng_jensen2017.py --levels 8` on a100server at commit cf16a7c, clean checkout (tmux `maesolvers-feng-jensen3`), 6 worker processes with one thread each: T39, T40 and T40 turned on `n = 10 2^l`, `l = 0, ..., 7`, with `m = 2, 4, ..., 64`, 141 solves.
The runner leaves out three solves at `n = 1280` with small `m` (`EXCLUDED`, with the reason in its docstring), where Howard's linear solves are slowest (Conventions, Linear solver; #72): T40 at `m = 2`, T40 turned at `m = 2` and 4.
None of them can be the best `m` at its level, so the tables below do not depend on them: T40 is at the rounding level for every `m`, and the run at dde1578 below solved T40 at `m = 2` to 8.0e-13 in 2 steps (MLflow run 99394efcf4db4f45b2a37c24e874cb94, 2.5 hours) and T40 turned at `m = 4` to 3.954e-2 in 17 steps (run 49e68b077354419b8f261c7f03b66ca0, 3.6 hours), against 5.76e-3 at `m = 64`; T40 turned at `m = 2` was never solved (more than 13 hours, stopped).
The runs of cf16a7c for these three rows were stopped, and MLflow marks them `KILLED`, as it does the six runs of the first runner at 119b73f in `-T39` and the unfinished `m = 2` run of dde1578 in `-T40_turned`.
A first run of the same scheme code at dde1578 (tmux `maesolvers-feng-jensen2`) gave the same errors, to the four printed digits, in all 141 shared solves; its rows existed only in memory when it was stopped, which is why the runner now writes each row as it finishes.
Every solve in the receipt converged (Howard's step below `1e-6`).
Every solve is an MLflow run named `n=<n> m=<m>` in `maesolvers-feng_jensen2017-T39`, `-T40` and `-T40_turned`, status `FINISHED`; the tag `mlflow.source.git.commit` tells the two runs apart.
The receipt's `nonconvex_fraction` column measures the least-squares Hessian fit, not the solution (Finding 3).
Diagnostics behind Finding 3 and the timing of `Newton` (Howard's algorithm): `feng_jensen2017_checks.csv`, written by `run_feng_jensen2017.py --checks` on a100server at commit 87566d9, clean checkout (`just remote`, 4 threads), each row also an MLflow run named `check=<kind> n=<n> m=<m>` in `maesolvers-feng_jensen2017-<problem>` (T20's in `-T20`).
`python benchmarks/run_feng_jensen2017.py --summary benchmarks/feng_jensen2017.csv` prints the tables below from the CSV.

## The comparison

The paper refines a quasi-uniform unstructured triangulation of the domain, from 91 nodes to 1,214,465 in 7 uniform refinements, and for each level prints the smallest relative error over `m` in `{2, 4, 8, 16, 32, 64}` with the minimising `m`.
We refine the grid `n = 10, 20, ..., 1280` cells per side of `[-1, 1]^2`, `h = 2/n`, which holds 1.14 to 1.43 times as many nodes as the paper's level of the same index (130 to 1.38 million), over the same `m`, and pair the levels by their index.
The errors are at the nodes, relative to `max |u|` over the domain: 4 for T39 (at the corner `(1, 1)`), 1 for T40.

## T39, the quartic problem: figure only

| n | nodes | ours | m | steps | paper's DoFs | paper | paper's m |
|---|---|---|---|---|---|---|---|
| 10 | 130 | 5.72e-2 | 2 | 4 | 91 | 9.26e-2 | 2 |
| 20 | 459 | 2.70e-2 | 2 | 4 | 329 | 3.85e-2 | 2 |
| 40 | 1,618 | 1.57e-2 | 4 | 5 | 1,249 | 2.15e-2 | 2 |
| 80 | 5,952 | 7.61e-3 | 4 | 5 | 4,865 | 9.59e-3 | 4 |
| 160 | 22,666 | 4.09e-3 | 8 | 5 | 19,201 | 5.63e-3 | 4 |
| 320 | 88,329 | 2.12e-3 | 8 | 5 | 76,289 | 2.44e-3 | 8 |
| 640 | 348,562 | 1.15e-3 | 16 | 5 | 304,129 | 1.51e-3 | 8 |
| 1280 | 1,384,524 | 6.71e-4 | 16 | 5 | 1,214,465 | 6.59e-4 | 16 |

Rates: ours `h^0.92` end to end, the paper's measure (`h^0.92` by least squares too), against `h^1.02`; in the best `k`, ours `k^1.60` against `k^1.78`.

Verdict: figure only, and close.
At every level ours lies between 0.62 and 1.02 times the paper's error, the best `m` follows the paper's to within a factor 2 (equal at 5 of 8 levels), and the end-to-end rate is 0.10 below the paper's.
By the numbers this would meet the order-of-magnitude level of ADR 0002 (ratio in `[1/2, 2]`, orders within 0.2), but that level compares at the paper's printed grids, and our grids are not the paper's: they are paired with its levels by index, with 14% to 43% more nodes (Conventions).
Each column of the paper's Figure 7 is ours too: at a fixed `m` the error stops falling once `h / k` is no longer small (at `m = 2` it levels off at 1.66e-2, the paper's at about 1.3e-2), and a larger `m` wins at a finer grid.

## T40, the nonsmooth problem: mismatch

Every solve, at every `n` and `m`, ends within 1e-12 of the exact solution at the nodes (largest relative error 5.4e-13, at `n = 1280`, `m = 4`; the run at dde1578 logged 8.0e-13 at `m = 2`), against 1.03e-1 to 4.51e-3 in the paper.
That is a mismatch in the direction of a better result, and Finding 1 explains it: it is a property of the structured grid, not of the scheme.

## Finding 1: on a grid whose columns follow the kink, the interpolant of T40 solves the scheme

T40's kink is the line `x1 = 0`.
Away from the boundary our triangulation is the lattice's, whose vertical lines through nodes are unions of triangle edges, and `x1 = 0` is either such a line (even `n`) or runs through the middle of a column of cells (odd `n`).
In both cases the piecewise linear interpolant `I u` of `u = |x1|` is there a convex function of `x1` alone, so along the stencil direction `(0, 1)` its second difference is 0, and along every other direction it is at least 0.
The frame of `(0, 1)` and `(1, 0)` then gives `-min(d1, d2) = 0`, every other frame gives at most 0, and `I u` solves the scheme, whose solution is unique.
The cells cut by the circle, whose triangles join lattice nodes to boundary points, are not covered by this argument; the receipt shows that they do not change the outcome: the error at the nodes is at the rounding level at every `n` and `m`.
`tests/test_semilagrangian.py` checks this at `n = 15` and 16.
The paper's mesh is unstructured, and the authors note that the kink "is not aligned with the computational mesh"; there `I u` is not convex along `(0, 1)` near the kink, and the scheme's error is `O(1e-2)`.

## Finding 2: turned off the grid, the nonsmooth problem behaves like the paper's

`T40 turned` is T40 with the kink along the line through 0 normal to `nu = (cos 0.3, sin 0.3)`: `u = |x . nu|`, `f = 0`, the Dirichlet data from `u`, `max |u| = cos 0.3 + sin 0.3` at the corner `(1, 1)`.
No grid line and no stencil direction is parallel to the kink (`0.3 + pi/2` is not a multiple of `pi/64`).

| n | nodes | ours | m | steps | paper (T40) | paper's m |
|---|---|---|---|---|---|---|
| 10 | 130 | 5.25e-2 | 8 | 4 | 1.03e-1 | 4 |
| 20 | 459 | 1.03e-2 | 16 | 4 | 5.69e-2 | 4 |
| 40 | 1,618 | 9.92e-3 | 32 | 4 | 3.08e-2 | 8 |
| 80 | 5,952 | 6.93e-3 | 64 | 5 | 2.03e-2 | 16 |
| 160 | 22,666 | 6.11e-3 | 64 | 6 | 1.38e-2 | 32 |
| 320 | 88,329 | 6.06e-3 | 64 | 8 | 9.12e-3 | 32 |
| 640 | 348,562 | 5.85e-3 | 64 | 8 | 6.04e-3 | 64 |
| 1280 | 1,384,524 | 5.76e-3 | 64 | 8 | 4.51e-3 | 64 |

Rates: ours `h^0.46` end to end and `h^0.34` by least squares, against the paper's `h^0.64` on T40: the end-to-end slope rests on the coarsest level, and from `n = 80` on our best error falls by only 17%.

At a fixed `m` the error hardly falls with `h` (at `m = 2`, 9.7e-2 from `n = 40` on; at `m = 64`, from 6.9e-3 to 5.8e-3 between `n = 80` and 1280): the error near a kink is set by the stencil size, of order `h^2 / k^2` times a second derivative that the kink makes of order `1 / h`.
So, as in the paper, a finer grid helps only with a wider stencil, and the best `m` grows with the refinement, from 8 to 64; the set of `m` stops at 64, which is best from `n = 80` on, so our best error stalls near 6e-3 where the paper's still falls.
At the finest two levels, where the paper's best `m` is 64 too, ours is within a factor 1.3 of its T40 error (5.85e-3 against 6.04e-3, 5.76e-3 against 4.51e-3); at the first six levels ours is 1.5 to 5.5 times smaller; there the wide stencils, clipped at the boundary, already do well on our grids, where the paper's errors at `m = 64` are about 0.24 on its first three levels (its Figure 7).
This is a comparison with a different problem, not a reproduction; it shows that the nonsmooth behaviour of Figure 7 comes from a kink across the mesh, which T40 on our grid lacks.

## Finding 3: the receipt's non-convex fraction measures the Hessian fit, not the solution

`node_hessian`, from which `evaluate` takes `nonconvex_fraction` and `eigmin`, is the least-squares fit of a symmetric matrix to the second differences along all 64 stencil directions (`fit_hessian`).
At a root of the scheme every second difference is nonnegative (the scheme's root conditions), but a fit to nonnegative data can be indefinite: one unit difference and the others 0 fit the eigenvalues `-1/D` and `3/D` for `D` directions (`tests/test_semilagrangian.py`).
The fit is worst where the differences are nearly all 0, in a flat region or next to a kink, so the receipt's `nonconvex_fraction` is 0.46 to 0.76 on T40 for solutions exact to 1e-12 (0.31 to 0.78 on T40 turned, 0 to 0.014 on T39).
`run_feng_jensen2017.py --checks` measures it at the root (`feng_jensen2017_checks.csv`, rows `fit`, commit 87566d9, 4 threads):

| problem | n | m | max error (absolute) | nonconvex_fraction | eigmin | nodes with all `d >= -1e-8` | of those, fitted eigenvalue `< -1e-6` | smallest `d` |
|---|---|---|---|---|---|---|---|---|
| T20 | 361 | 4 | 1.06e-3 | 8.9% | -2.52e-2 | 95.2% | 4.0% (down to -2.49e-2) | -8.1e-4 |
| T39 | 320 | 8 | 8.47e-3 | 0 | 1.1e-4 | 99.999% | 0 | -5.4e-7 |
| T40 | 80 | 16 | 1.0e-15 | 65% | -2.34 | 100% | 25% (down to -2.34) | -1.2e-10 |
| T40 | 320 | 16 | 5.8e-15 | 64% | -9.25 | 100% | 7.8% (down to -9.25) | -1.9e-9 |
| T40 turned | 320 | 16 | 2.16e-2 | 58% | -4.67 | 100% | 16% (down to -4.67) | -9.3e-10 |

The second differences are nonnegative up to rounding on T39 and T40, and on T20 down to -8.1e-4, a trace of Howard's stopping tolerance (`1e-6` in `u`, divided by `k^2 / 2`); the fit's negative eigenvalues are 30 times larger there and of order 1 on T40.
The Hessian of the maximising frame, `d1 s1 s1^T + d2 s2 s2^T`, has the eigenvalues `d1`, `d2`, so its smallest eigenvalue is the smallest `d` of that frame: the same values as the last column here.
Reporting it as `node_hessian` (#73) would change the `nonconvex_fraction` and `eigmin` columns of this receipt and of `comparison.csv`, so it is left for that issue.
The T20 row's 8.9% differs from the comparison receipt's 8.8%, solved with one thread: the rounding of the Krylov solves depends on the thread count, and these two metrics are sensitive to it (the max error agrees to six digits).

## Howard's algorithm

Fig. 8 of the paper gives 4 to 16 steps to a step below `1e-6` over all levels and `m`, 5 to 8 (quartic) and 5 to 9 (nonsmooth) at the best `m`.
Ours, over all levels and `m` of the receipt: 4 to 7 on T39 (4 or 5 at the best `m`), 2 to 9 on T40, and 4 to 27 on T40 turned, the most at `m = 2` and 4, where the narrow stencil sees the kink; at the best `m`, 4 to 8.

`Newton(scheme=SemiLagrangian(n))` reaches the same root, but `Newton` solves its linear systems by LU, which fills in on wide stencils: on T39 it took 0.85 s against 0.37 s for `FDSemiLagrangian` at `n = 64`, `m = 4`, and 64.1 s against 1.3 s at `n = 128`, `m = 8`, with the same error to 1e-10 (`feng_jensen2017_checks.csv`, rows `newton` and `howard`, 4 threads).
#72 proposes to let the scheme choose its linear solve.

## Conventions that differ

- **Grid.** The paper's nodes are those of a quasi-uniform unstructured triangulation refined uniformly; ours are the grid nodes in the open domain and the boundary points of `shortley_weller_operators`, triangulated by Delaunay (`docs/methods/feng_jensen2017.md`). Our grids hold 1.14 to 1.43 times the paper's number of nodes at the same level index.
- **Mesh size.** The paper's `h` is the "(average) mesh size" and `k = m h`; ours is the grid spacing `2/n`. The paper does not print its `h`, so `k` cannot be compared directly.
- **Stencil size near the boundary.** The paper shrinks the stencil where `m h` exceeds the distance to the boundary of the triangulated domain, and its Figure 5(b) shows a size that depends on the direction; ours is `min(m h, distance)` at each node, the same for every direction.
- **Split.** The paper discretises the rotations of the frames by `i pi / 64` and says nothing about the split `a` of `diag(a, 1 - a)`; we take the supremum over `a` in closed form, which any fine discretisation of `a` approaches.
- **Error measure.** Ours is the max error at the nodes relative to `max |u|` over the domain; the paper's `L^inf` error is that of the finite element function, whose maximum over the domain is at most the nodal one plus the interpolation error, `O(h^2)` for T39.
- **Linear solver.** The paper used FEniCS without naming its linear solver; ours is BiCGSTAB with diagonal scaling, then GMRES, then LU (`Howard`). The discrete solution is the one an LU factorisation gives, but at small `m` on fine grids the Krylov methods need many steps, and their budget of `10 N` steps each keeps the LU fallback out of reach (Rerun; #72).

## Not compared

- The `L^2` and `H^1` columns of Figure 6 and their rates: `evaluate` has no such fields (ADR 0007), and the paper does not say how it integrates the errors.
- Figure 4 (the solutions on the coarsest mesh).

## Rerun

```sh
just push
just remote-bg feng-jensen env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/run_feng_jensen2017.py --processes 6 --levels 8 \
    --out /jczhang02/maesolvers-data/data/feng_jensen2017.csv
# when the log ends:
scp a100server:/jczhang02/maesolvers-data/data/feng_jensen2017.csv benchmarks/
uv run python benchmarks/run_feng_jensen2017.py --summary benchmarks/feng_jensen2017.csv
# the checks of Finding 3 and of Newton's LU, about 3 minutes:
just remote env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python benchmarks/run_feng_jensen2017.py --checks \
    --out /jczhang02/maesolvers-data/data/feng_jensen2017_checks.csv
scp a100server:/jczhang02/maesolvers-data/data/feng_jensen2017_checks.csv benchmarks/
```

The 141 solves end; those at `n = 1280` take 7 to 47 minutes each at `m >= 4` and 2 hours for T39 at `m = 2`; each worker at `n = 1280` holds about 10 GB.
The three excluded solves take hours (Receipt), because BiCGSTAB needs thousands of steps and often breaks down into GMRES, whose budget of `10 N` steps keeps the LU fallback out of reach (#72).
The 141 rows took 2.5 hours with 6 workers.
A rerun with the same `--out` solves only the rows it lacks.
The checks run their solves one after the other, in about 3 minutes; each is an MLflow run named `check=<kind> n=<n> m=<m>` in `maesolvers-feng_jensen2017-<problem>`, T20's in `maesolvers-feng_jensen2017-T20`.
