# Comparison table: every preset on the shared test set

The comparison table (GLOSSARY: 对比表) runs every preset of `maesolvers.SOLVERS` with its defaults on the shared test set T1, T2, T20, T8, T12, T9, T10, and measures each run with `evaluate` (ADR 0007).
It carries no published numbers; those are in the reproduction tables (`standard_fd.md`, `monotone17.md`, `monotone19.md`, `hybrid.md`, `filtered.md`, `multigrid.md`, `bfo2014.md`, `chen_wan_lin2018.md`, `feng_jensen2017.md`, `penalty_mlp.md`, `icnn.md`).
README's section "Comparison" shows the tables; this file says how they were made and what they show.

## Receipt

- `comparison.csv`, one row per run: one per classical preset and problem, one per neural preset, problem and seed, and one row with status `unsupported` and the reason for each pair a preset does not support (199 runs and 52 unsupported pairs).
  The columns are the fields of `evaluate`, the run's `converged`, `stop` (neural), `device`, `threads`, `cpu` (the CPU's model name), `commit` and `hparams` (the preset's `hparams` as JSON); `comparison.py` lists them and the statuses.
- Written on a100server by `run_comparison.py` in one run of every preset at commit a717d42 on the branch of PR #76 (issue #34), clean checkout, in 5006 s of wall time: `just remote-bg comparison-0-2 env CUDA_VISIBLE_DEVICES= python benchmarks/run_comparison.py --out /jczhang02/maesolvers-data/data/comparison-0.2.csv`, then copied here; its log is `/jczhang02/maesolvers-data/logs/comparison-0-2.log` on a100server.
  The runs took 16 worker processes, each with one CPU thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and torch's threads set to 1), neural runs first, and the machine was shared with other jobs (load averages of 8 to 26 where sampled during the run).
- It replaces a receipt that was merged from runs of single presets at seven commits: f15b98f for the presets of 0.1 (PR #58), 21970e5 for `TransportICNN` (#66), 2ba8083 for `FDTransport` (#67), 3d31695 for `FDMultigrid` (#68), 249e131 for `DirichletQSPINN` and `TransportQSPINN` (#69), 55da491 for `FDMixed` (#70) and dde1578 for `FDSemiLagrangian` (#71).
  Those branches were squash-merged, so the commits are not in the history of `next`; GitHub keeps them under `refs/pull/<number>/head`, and the receipt itself is `benchmarks/comparison.csv` at b9ca39e.
  Every field of all 251 rows but `time` and `commit` agrees with it to every digit, `hparams` and the failed seed included, so no preset changed its results between its own branch and the integration of 0.2.
  The times are 0.57 to 1.37 times the old ones (median 0.91): the run differs in the load of the shared server, not in the work (`FDMultigrid` on T9, run alone before, takes 163 s here against 155 s).
- Every run is one MLflow run in the experiment `maesolvers-<preset in lower case>-<problem>` (for example `maesolvers-fdnewton-T1`, `maesolvers-dirichleticnn-T9`), named `comparison n=361` or `comparison seed=<s>` and tagged `benchmark=comparison`; the neural runs carry their training curves.
  The runs of the earlier receipts are in the same experiments under the same names, told apart by their commit tag.
- `comparison.py` writes README's tables from the CSV (`just comparison-table`), with notes derived from the rows (the grid steps, the seeds, the machine, the reasons), and `tests/test_comparison.py` checks that README is current and that the CSV covers every preset on every problem of the shared test set, with seeds 0 to 9 for a neural preset, `n = 361` for a classical one, the reason `unsupported_reason` gives for an unsupported pair, and the `hparams` the preset has now.
  A new preset, a preset whose support or defaults change, fails that test until `run_comparison.py` runs again.

## Settings

- Classical presets: one run on `n = 361` cells per side, the largest `N` the papers print in 2D (test-set specification, Section 2; ADR 0001), so `h = 1/361` on the unit square and `h = 2/361` on the disk of T8 and on the square `[-1, 1]^2` in which `FDTransport` embeds the disk of T12.
  The error is taken at the preset's own nodes, the literature's practice.
- Neural presets: seeds 0 to 9 with every other setting at the preset's default (ADR 0008); the error is taken on the shared evaluation grid (201 x 201 in 2D, 51^3 for T9, 21^4 for T10).
  A cell gives the median and the 5% to 95% band of the max error over the seeds, then the number of successful seeds.
- Success threshold: a seed succeeds when its max error is below 1% of the exact solution's oscillation `max u - min u` on the shared grid (`RELATIVE_THRESHOLD` in `comparison.py`): 1.72e-2 on T1, 1.41e-2 on T2, 1.29e-3 on T20, 5.00e-3 on T8, 6.93e-2 on T12, 3.48e-2 on T9, 6.39e-2 on T10.
  ADR 0007 leaves the threshold to the caller; a relative one gives the same meaning to every row, where the solutions' oscillations differ by a factor of 50 (0.129 on T20, 6.39 on T10).
  The oscillation does not change when a constant is added to `u`, which the test-set specification says no metric may see (T8's `g = 0` against `1/2`), and it is defined for T12, whose solution is unique only up to a constant; `max |u|` has neither property.
- Failed runs: a run that raises, has a non-finite error, or (neural) stops as `"diverged"` is a failed run, with error `+inf` in the band (ADR 0007, and `summarize` since this change).
  One run of this receipt failed: `DirichletQSPINN` on T2, seed 9, stops `"diverged"` after 190 steps (Finding 4).
- A classical run whose iteration stops before its tolerance (`converged` False) is kept at its last iterate and marked `\*`; a neural seed that stops at its step budget or stalls is a normal seed.
- Unsupported pairs: `unsupported_reason` in `comparison.py` names the first of the boundary condition, the dimension and the domain that the preset does not declare.
  No preset of 0.1 takes T12's transport boundary condition (`Dirichlet only`); in 0.2 two presets take only transport problems (`Transport only` on the other rows), `FDTransport` (#28) and `TransportICNN` (#29), and later `TransportQSPINN` (#33); the other presets, finite-difference and neural, stay `Dirichlet only`.
  On T12 the monotone start of `FDTransport` stalls (no step decreases its residual; the `RuntimeWarning` "FDTransport.start did not converge on T12" in the run's log, `/jczhang02/maesolvers-data/logs/comparison-0-2.log` on a100server, as `comparison.csv` has no column for it) and its Newton iteration on the filtered scheme stops at step 142 of 200, where no step decreases the max residual, so the cell is marked `\*` (`bfo2014.md`, Finding 2, and #56); it supports no other problem of the set (`Transport only`).
  For a preset on the QS form it also names the reference point when that is what fails: `DirichletQSPINN` on T20 gives `f(x0) <= 0`, since T20's `f` vanishes at the centre of its box (`QuadraticForm.refusal` gives the reason) (`docs/methods/qs_pinn.md`, Known limits).
- Orientation: the problems are the rows and the presets the columns, as GLOSSARY (共用集) and the test-set specification say; the scope comment of #14 calls T12 a "column".

## Differences from the test-set specification

- Device: the specification asks for neural times on one GPU; every neural run here trained on the CPU, one thread per seed, the presets' default, because the networks are small and one A100 was no faster than one CPU thread (`penalty_mlp.md`, Device).
  So the time table compares one CPU thread with one CPU thread.
- The specification also asks for `rms`, `eigmin` and `eigmin_exact` in the appendix tables; they are in the CSV, not in README.
- The specification asks for the iterations and the unknowns of a classical run next to its time; they are in the CSV (columns `iterations` and `unknowns`), not in README, and Finding 6 gives the unknowns.

## Agreement with the reproduction receipts

Every value the reproduction receipts share with this one agrees to all digits, so the presets did not move between those receipts and this one:

- `FDNewton` on T1 (5.276e-7) and T2 (2.225e-2), `FDPoisson` on T2 (5.409e-3): `standard_fd.md`, `N = 361`.
- `FDMonotone` on T1 (4.400311e-4, 14 steps), T2 (3.261801e-4, 30 steps) and T20 (6.602350e-4, 43 steps): `monotone17.csv`, `N = 361`, `h = 1/N`, 17 points, narrow.
- `FDHybrid` on T1 (1.571050e-6), T2 (4.177276e-5) and T20 (2.970804e-5): `hybrid.csv`, `N = 361`, `h = 1/N`, narrow.
- `FDFiltered` on T1 (9.677519e-5, 14 steps, `converged` False): `filtered.csv`, `N = 361`, `h = 1/N`, 17 points, narrow; its T2 and T20 rows there are at `h = 1/(N-1)` only.
- `PenaltyMLP` and `DirichletICNN` on T1, seeds 0 to 9: `penalty_mlp_t1.csv` and `icnn_t1.csv`, every seed's shared-grid max error.
- `DirichletQSPINN` on T8 and `TransportQSPINN` on T12, seeds 0 to 9: `qs_pinn.csv`, every seed's shared-grid max error.

One receipt differs by design: `icnn_t12.csv` trained `TransportICNN` on T12 on a GPU, and this table on the CPU, the preset's default.
The two devices round differently, and 1800 L-BFGS steps carry the difference into the result: the CPU seeds' max error of `u` is 0.79 to 1.80 times the GPU seeds' (median 1.7e-3 here, 2.0e-3 there), the same distribution, not the same numbers.

## What the table shows

### Finding 1: T8 does not separate the standard scheme's iterations by accuracy

The exact solution of T8, `(|x|^2 - 1) / 2`, is quadratic, and `FDNewton` and `FDPoisson` reach it to rounding at the nodes (3.3e-13 and 1.6e-12): on this `u` the discrete equations of `Standard9`, the Shortley-Weller stencils next to the circle included, hold exactly.
Their secondary value on the shared grid, 2.61e-6, is the error of the Clough-Tocher interpolant, not of the scheme.
Both reach the convex branch (non-convex fraction 0), from the Poisson start; T8 tests branch selection, and on it neither iteration picks the concave solution.
The monotone, hybrid and filtered schemes are box-only in 0.1 (`not on a Disk`); the semi-Lagrangian scheme, monotone too, runs on the disk, with 9.0e-6 at the nodes, the error of its piecewise linear interpolation of the quadratic `u` at the stencil ends (Finding 9).

### Finding 2: `FDFiltered` stops before its tolerance on T1, T2 and T20

At `n = 361` the damped Newton iteration of `FDFiltered` stops after 14, 21 and 56 steps with `converged` False on all three box problems, the known limit of `filtered.md` (Finding 4, #56); its cells are the last iterates and carry `\*`.

### Finding 3: every classical preset and `PenaltyMLP` have non-convex points on T20; `DirichletICNN` has none

On T20, `f = 0` on the disk `|x - x0| < 0.2`, where the exact Hessian is 0; the centred discrete Hessian of every classical preset has a negative smallest eigenvalue at 3.4% (`FDPoisson`) to 10.9% (`FDMixed`) of the interior nodes (`FDMultigrid` 5.7%, after cycles that do not converge, Finding 7), as `monotone17.md` (Finding 6) and `filtered.md` record for single presets.
`DirichletICNN` is convex by construction (0 on every problem); `PenaltyMLP` penalises only negative `u_xx` and `u_yy` and has non-convex points on T1, T2 and T20.
On T2, `FDNewton` reaches its known non-convex root (1.09% of the nodes, `standard_fd.md`, Finding 2); the monotone, hybrid and filtered schemes have 10 non-convex nodes (0.0077%), while `FDPoisson` and `FDMultigrid` have none (`FDMultigrid` had 6 when it stopped after two cycles, `multigrid.md`, Finding 4).
In one solve of `FDMonotone` outside this receipt, not recorded, these nodes lay within four grid steps of the corner `(1, 1)`, where the gradient blows up.
`FDSemiLagrangian`'s 8.8% on T20 (smallest eigenvalue -2.5e-2) measures its Hessian fit, not its solution: its discrete Hessian is the least-squares fit of a symmetric matrix to its second differences along 64 directions, and a fit to nonnegative differences can be indefinite (one unit difference and the others 0 fit the eigenvalues `-1/64` and `3/64`), worst where they are nearly all 0, in the flat disk and next to kinks.
At its root every second difference is at least -8.1e-4 (a trace of Howard's stopping tolerance, `1e-6` in `u` divided by `k^2 / 2`), and at the 95.2% of the nodes where all of them are at least `-1e-8`, 4.0% still have a fitted eigenvalue below `-1e-6`, down to -2.5e-2 (`feng_jensen2017_checks.csv`, row `fit` T20, a solve with 4 threads whose fraction is 8.9%, the rounding of the Krylov solves differing; `feng_jensen2017.md`, Finding 3).
So its cell in the non-convex column is not comparable with the other classical presets' centred Hessians; #73 proposes the Hessian of the maximising frame instead.

### Finding 4: no neural seed meets the threshold on T2 and T20

On T2 the penalty MLP's median max error is 0.121 (8.6% of the oscillation), and 9 of 10 `DirichletICNN` seeds stall (SciPy's line search fails) after 509 to 7650 steps, with errors from 0.066 to 2.2e3; only seed 3 reaches the step budget, at 0.049.
On T20, whose solution is only `C^1`, the medians are 4.3e-3 (penalty MLP) and 1.4e-2 (ICNN) against a threshold of 1.3e-3.
On T1 and T8, which are smooth, `DirichletICNN` succeeds in every seed (on T8 eight seeds meet the gradient tolerance), and the penalty MLP in 9 (seed 1 ends at 1.79e-2, just above the threshold of 1.72e-2) and 5.
`DirichletQSPINN` fails the threshold on T2 in every seed too (median 1.80, errors 0.34 to 5.1; nine seeds `stalled` after 1 to 1035 steps, and seed 9 diverges after 190 steps, a failed run; non-convex at 18.7% of the grid in the median), and does not support T20, whose `f` vanishes at the centre of its box (`f(x0) <= 0`); on T1 and T8 it succeeds in every seed, on T8 at 1.72e-7, which `qs_pinn.md` (Finding 1) explains.

### Finding 5: `DirichletICNN` in 3D works in 7 seeds of 10, in 4D in none

On T9 (3D) eight seeds reach the step budget, seven of them below the threshold of 3.48e-2 (3.1e-3 to 2.9e-2) and seed 0 above it (9.2e-2); seed 4 stalls after 1843 steps at 0.83 and seed 9 after 3 steps at 1.9e3.
On T10 (4D) nine seeds stall, four of them after one step (errors 1.6e4 to 4.8e4), and the best seed ends at 2.5; `docs/methods/icnn.md` (Known limits) records first losses of 2e33 to 3e36 there, against 2e26 to 3e28 on T9.
The time median on T10 (146 s) is short because most seeds stop early, not because the solve is cheap.
`DirichletQSPINN` succeeds in every seed on T9 (4.2e-3 to 1.7e-2, every seed `stalled`), and in none on T10, where every seed stalls after 645 to 1359 steps at errors of 0.73 to 1.45 and five are non-convex at some grid points.
Its default `ratio = 0.6` is too small for the canonical QS representation of T10, which needs `|A / alpha - I|_F` up to 0.64 (0.46 for T9, 0.28 for T1), but that does not set the error: with `ratio = 0.9` every seed still stalls, at 9.3e-1 in the median (`qs_pinn.md`, Finding 5).

### Finding 6: cost

At `n = 361` the classical presets take 0.77 s (`FDMultigrid` on T1) to 258 s (`FDMixed` on T20) on one thread, with 129600 unknowns on the square and 102328 on the disk, and `FDMultigrid` 163 s on T9, with 46656000 unknowns; the neural presets take 345 s to 1186 s per seed (medians), except where their seeds stop early: `DirichletICNN` on T10 (146 s, Finding 5), `DirichletQSPINN`, whose seeds stall or converge sooner (40 s to 398 s), and the transport presets on T12 (171 s and 177 s for their 2000 steps).
On T1, T2 and T20 the neural presets' median errors are two to four orders of magnitude above the best classical preset's.
The cheapest accurate classical preset differs by problem: `FDPoisson` on T1 (2.5 s, 5.28e-7), `FDHybrid` on T2 (4.18e-5 in 129 s, against 5.41e-3 for `FDPoisson`), `FDPoisson` on T20 (1.59e-5 in 33 s).

### Finding 7: `FDMultigrid` is the one classical preset whose defaults run T9, and the fastest on the square; it does not converge on T20

`FDMultigrid` solves T9 at `n = 361` (`361^3` nodes) in one full multigrid cycle, with a max error of 9.09e-7 and no non-convex node.
On the square it stops after one cycle on T1 (0.77 s) and nine on T2 (3.6 s); its errors are those of the standard scheme it solves, close to `FDPoisson`'s: 6.81e-7 against 5.28e-7 on T1 (one cycle stops above the discrete solution, `multigrid.md`, Finding 1) and 5.41e-3 against 5.41e-3 on T2.
On T20, where `f = 0` on a disk, its cycles stall: it runs its 50 cycles in 19 s and reports `converged` False, so its cell carries `\*`, with an error of 1.69e-5 (against 1.59e-5 for `FDPoisson`) and 5.7% non-convex nodes; the paper warns that the method may break down on degenerate data (`docs/methods/multigrid.md`, Known limits).
The first receipt of these rows (fe47d9b) stopped by the paper's relative residual, which holds after one cycle on fine grids whatever the iterate (`multigrid.md`, Findings 2 and 5): it reported one cycle and `converged` True on T20 (3.51e-5, 0.72% non-convex nodes) and two cycles on T2 (5.40e-3, 6 non-convex nodes).
`n = 361` is not a power of 2, so its grids (361, 181, 91, 46, 23, 12, 6, 3, 2 cells) are not nested and the transfers interpolate linearly between them (`docs/methods/multigrid.md`); the paper's grids are powers of 2.

### Finding 8: `FDMixed` on T2 ends on its second root

`FDMixed` reaches 7.0e-7 on T1 with the 7-point stencil at every node, but 2.13e-4 on T2, the second root of `chen_wan_lin2018.md`, Finding 5: from the paper's start, policy iteration ends with the wide stencil at a few nodes next to the corner `(1, 1)` (28 policy iterations; 18 non-convex nodes, 0.014%), as it does there from `N = 256` on; its 7-point root, which it reaches from the exact solution, is more accurate.

### Finding 9: `FDSemiLagrangian` at its default stencil factor is accurate to about 1e-3

At `n = 361` and `m = 4`, `FDSemiLagrangian` reaches 3.9e-3 on T1, 2.2e-3 on T2 and 1.1e-3 on T20, in 4 to 8 steps of Howard's algorithm and 20 s to 31 s; on T8, 9.0e-6 in 3 steps and 17 s.
The error is that of the interpolation at the stencil ends, of order `(h / k)^2 = 1/16` times the second derivatives, which does not fall with `h` at a fixed `m`; `feng_jensen2017.md` shows that the paper's best `m` grows with the refinement for the same reason.
On T2 it is the third most accurate of the classical presets that converge, after `FDHybrid` and `FDMonotone`, and it is the only monotone preset that runs on the disk of T8.

## Rerun

```sh
just push
just remote-bg comparison python benchmarks/run_comparison.py \
    --out /jczhang02/maesolvers-data/data/comparison.csv
# when the log ends with "wrote ...":
scp a100server:/jczhang02/maesolvers-data/data/comparison.csv benchmarks/
just comparison-table
```

`--solvers` and `--problems` rerun some pairs and replace only their rows of the CSV given by `--out`; copy the current `comparison.csv` there first.
