# Semi-Lagrangian scheme: published versus ours

`FDSemiLagrangian` (`Howard` on `SemiLagrangian`) against Feng and Jensen, SIAM J. Numer. Anal. 55 (2017), Section 8, read from the journal PDF (`sources.md`, section 6).
**Levels (ADR 0002): figure only on T39**, against Figure 6, a table of the best relative error over the stencil factor `m` at each level under a figure caption, and the rate `h^1.02` of p. 711; **mismatch on T40**, which our structured grid solves to the rounding level (Finding 1).
The paper's nonsmooth behaviour is compared on T40 turned off the grid, a test-local problem without a T id (Finding 2).

Receipt: `feng_jensen2017.csv`, written by `run_feng_jensen2017.py` on a100server at commit cf16a7c, clean checkout (tmux `maesolvers-feng-jensen3`), 6 worker processes with one thread each: T39, T40 and T40 turned on `n = 10 2^l`, `l = 0, ..., 7`, with `m = 2, 4, ..., 64`, 141 of the 144 solves.
The three missing rows are at `n = 1280` with small `m`, where Howard's linear solves are slowest (Conventions, Linear solver): T40 at `m = 2`, T40 turned at `m = 2` and 4; none of them can be the best `m` at its level (T40 is at the rounding level for every `m`; T40 turned has 5.76e-3 at `m = 64` against 3.95e-2 at `m = 4`), so the tables below do not depend on them.
A first run of the same scheme code at dde1578 (tmux `maesolvers-feng-jensen2`) gave the same errors, to the four printed digits, in all 141 shared solves, and the `m = 4` row in its log (3.954e-2); its solve of T40 turned at `n = 1280`, `m = 2` was still running after 13 hours, and its rows existed only in memory, which is why the runner now writes each row as it finishes.
Every solve converged (Howard's step below `1e-6`).
Every solve is an MLflow run named `n=<n> m=<m>` in `maesolvers-feng_jensen2017-T39`, `-T40` and `-T40_turned`; the tag `mlflow.source.git.commit` tells the two runs apart.
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

## Howard's algorithm

Fig. 8 of the paper gives 4 to 16 steps to a step below `1e-6` over all levels and `m`, 5 to 8 (quartic) and 5 to 9 (nonsmooth) at the best `m`.
Ours, over all levels and `m` of the receipt: 4 to 7 on T39 (4 or 5 at the best `m`), 2 to 9 on T40, and 4 to 27 on T40 turned, the most at `m = 2` and 4, where the narrow stencil sees the kink; at the best `m`, 4 to 8.

## Conventions that differ

- **Grid.** The paper's nodes are those of a quasi-uniform unstructured triangulation refined uniformly; ours are the grid nodes in the open domain and the boundary points of `shortley_weller_operators`, triangulated by Delaunay (`docs/methods/feng_jensen2017.md`). Our grids hold 1.14 to 1.43 times the paper's number of nodes at the same level index.
- **Mesh size.** The paper's `h` is the "(average) mesh size" and `k = m h`; ours is the grid spacing `2/n`. The paper does not print its `h`, so `k` cannot be compared directly.
- **Stencil size near the boundary.** The paper shrinks the stencil where `m h` exceeds the distance to the boundary of the triangulated domain, and its Figure 5(b) shows a size that depends on the direction; ours is `min(m h, distance)` at each node, the same for every direction.
- **Split.** The paper discretises the rotations of the frames by `i pi / 64` and says nothing about the split `a` of `diag(a, 1 - a)`; we take the supremum over `a` in closed form, which any fine discretisation of `a` approaches.
- **Error measure.** Ours is the max error at the nodes relative to `max |u|` over the domain; the paper's `L^inf` error is that of the finite element function, whose maximum over the domain is at most the nodal one plus the interpolation error, `O(h^2)` for T39.
- **Linear solver.** The paper used FEniCS without naming its linear solver; ours is BiCGSTAB with diagonal scaling, then GMRES, then LU (`Howard`). The discrete solution is the same, but at small `m` on fine grids the Krylov methods need many steps (Rerun).

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
```

The solves at `n = 1280` take 7 to 47 minutes each at `m >= 4`, 2 hours for T39 at `m = 2`, and more than 13 hours for T40 turned at `m = 2`, where BiCGSTAB needs thousands of steps and often breaks down into GMRES; each worker at `n = 1280` holds about 10 GB.
The 141 rows took 2.5 hours with 6 workers.
A rerun with the same `--out` solves only the rows it lacks, so the three missing rows can be added by running the command again and stopping it when they are written.
