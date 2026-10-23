# Comparison table: every preset on the shared test set

The comparison table (GLOSSARY: 对比表) runs every preset of `maesolvers.SOLVERS` with its defaults on the shared test set T1, T2, T20, T8, T12, T9, T10, and measures each run with `evaluate` (ADR 0007).
It carries no published numbers; those are in the reproduction tables (`standard_fd.md`, `monotone17.md`, `hybrid.md`, `filtered.md`, `penalty_mlp.md`, `icnn.md`).
README's section "Comparison" shows the tables; this file says how they were made and what they show.

## Receipt

- `comparison.csv`, one row per run: one per classical preset and problem, one per neural preset, problem and seed, and one row with status `unsupported` and the reason for each pair a preset does not support (127 runs and 28 unsupported pairs).
  The columns are the fields of `evaluate`, the run's `converged`, `stop` (neural), `device`, `threads`, `cpu` (the CPU's model name), `commit` and `hparams` (the preset's `hparams` as JSON); `comparison.py` lists them and the statuses.
- Written on a100server by `run_comparison.py` at commit f15b98f, clean checkout, in 4853 s of wall time: `just remote-bg comparison-r1 python benchmarks/run_comparison.py --out /jczhang02/maesolvers-data/data/comparison.csv`, then copied here.
  It replaces the receipt of commit 5c423bc, which lacked the `cpu` and `hparams` columns; every field but `time` agrees with it to all digits, and the times differ by a factor of 0.31 to 1.23 (median 1.10) with the load of the shared server.
  The commits this file cites are on the branch of PR #58, which was squash-merged into `next` as 4cc6712, so they are not in its history; GitHub keeps them under `refs/pull/58/head`.
  The runs took 16 worker processes, each with one CPU thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and torch's threads set to 1), neural runs first.
- The rows of `TransportICNN` (issue #29) came later, from the same script at commit 21970e5, clean checkout, merged into the receipt in 496 s of wall time with 10 worker processes of one CPU thread each: `just remote-bg comparison-transport-icnn env CUDA_VISIBLE_DEVICES= python benchmarks/run_comparison.py --solvers TransportICNN --workers 10`.
  The merge replaces only that preset's rows, so every other row is the receipt of f15b98f.
- Every run is one MLflow run in the experiment `maesolvers-<preset in lower case>-<problem>` (for example `maesolvers-fdnewton-T1`, `maesolvers-dirichleticnn-T9`), named `comparison n=361` or `comparison seed=<s>` and tagged `benchmark=comparison`; the neural runs carry their training curves.
- `comparison.py` writes README's tables from the CSV (`just comparison-table`), with notes derived from the rows (the grid steps, the seeds, the machine, the reasons), and `tests/test_comparison.py` checks that README is current and that the CSV covers every preset on every problem of the shared test set, with seeds 0 to 9 for a neural preset, `n = 361` for a classical one, the reason `unsupported_reason` gives for an unsupported pair, and the `hparams` the preset has now.
  A new preset, a preset whose support or defaults change, fails that test until `run_comparison.py` runs again.

## Settings

- Classical presets: one run on `n = 361` cells per side, the largest `N` the papers print in 2D (test-set specification, Section 2; ADR 0001), so `h = 1/361` on the unit square and `h = 2/361` on the disk of T8.
  The error is taken at the preset's own nodes, the literature's practice.
- Neural presets: seeds 0 to 9 with every other setting at the preset's default (ADR 0008); the error is taken on the shared evaluation grid (201 x 201 in 2D, 51^3 for T9, 21^4 for T10).
  A cell gives the median and the 5% to 95% band of the max error over the seeds, then the number of successful seeds.
- Success threshold: a seed succeeds when its max error is below 1% of the exact solution's oscillation `max u - min u` on the shared grid (`RELATIVE_THRESHOLD` in `comparison.py`): 1.72e-2 on T1, 1.41e-2 on T2, 1.29e-3 on T20, 5.00e-3 on T8, 6.93e-2 on T12, 3.48e-2 on T9, 6.39e-2 on T10.
  ADR 0007 leaves the threshold to the caller; a relative one gives the same meaning to every row, where the solutions' oscillations differ by a factor of 50 (0.129 on T20, 6.39 on T10).
  The oscillation does not change when a constant is added to `u`, which the test-set specification says no metric may see (T8's `g = 0` against `1/2`), and it is defined for T12, whose solution is unique only up to a constant; `max |u|` has neither property.
- Failed runs: a run that raises, has a non-finite error, or (neural) stops as `"diverged"` is a failed run, with error `+inf` in the band (ADR 0007, and `summarize` since this change).
  No run of this receipt failed.
- A classical run whose iteration stops before its tolerance (`converged` False) is kept at its last iterate and marked `\*`; a neural seed that stops at its step budget or stalls is a normal seed.
- Unsupported pairs: `unsupported_reason` in `comparison.py` names the first of the boundary condition, the dimension and the domain that the preset does not declare.
  No preset of 0.1 takes T12's transport boundary condition (`Dirichlet only`); `TransportICNN`, the first transport preset (0.2), takes only transport problems (`Transport only` on the other rows).
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

One receipt differs by design: `icnn_t12.csv` trained `TransportICNN` on T12 on a GPU, and this table on the CPU, the preset's default.
The two devices round differently, and 1800 L-BFGS steps carry the difference into the result: the CPU seeds' max error of `u` is 0.79 to 1.80 times the GPU seeds' (median 1.7e-3 here, 2.0e-3 there), the same distribution, not the same numbers.

## What the table shows

### Finding 1: T8 does not separate the standard scheme's iterations by accuracy

The exact solution of T8, `(|x|^2 - 1) / 2`, is quadratic, and `FDNewton` and `FDPoisson` reach it to rounding at the nodes (3.3e-13 and 1.6e-12): on this `u` the discrete equations of `Standard9`, the Shortley-Weller stencils next to the circle included, hold exactly.
Their secondary value on the shared grid, 2.61e-6, is the error of the Clough-Tocher interpolant, not of the scheme.
Both reach the convex branch (non-convex fraction 0), from the Poisson start; T8 tests branch selection, and on it neither iteration picks the concave solution.
The monotone, hybrid and filtered schemes are box-only in 0.1 (`not on a Disk`).

### Finding 2: `FDFiltered` stops before its tolerance on T1, T2 and T20

At `n = 361` the damped Newton iteration of `FDFiltered` stops after 14, 21 and 56 steps with `converged` False on all three box problems, the known limit of `filtered.md` (Finding 4, #56); its cells are the last iterates and carry `\*`.

### Finding 3: every classical preset and `PenaltyMLP` have non-convex points on T20; `DirichletICNN` has none

On T20, `f = 0` on the disk `|x - x0| < 0.2`, where the exact Hessian is 0; the centred discrete Hessian of every classical preset has a negative smallest eigenvalue at 3.4% (`FDPoisson`) to 9.8% (`FDNewton`) of the interior nodes, as `monotone17.md` (Finding 6) and `filtered.md` record for single presets.
`DirichletICNN` is convex by construction (0 on every problem); `PenaltyMLP` penalises only negative `u_xx` and `u_yy` and has non-convex points on T1, T2 and T20.
On T2, `FDNewton` reaches its known non-convex root (1.09% of the nodes, `standard_fd.md`, Finding 2); the monotone, hybrid and filtered schemes have 10 non-convex nodes (0.0077%), while `FDPoisson` has none.
In one solve of `FDMonotone` outside this receipt, not recorded, these nodes lay within four grid steps of the corner `(1, 1)`, where the gradient blows up.

### Finding 4: no neural seed meets the threshold on T2 and T20

On T2 the penalty MLP's median max error is 0.121 (8.6% of the oscillation), and 9 of 10 `DirichletICNN` seeds stall (SciPy's line search fails) after 509 to 7650 steps, with errors from 0.066 to 2.2e3; only seed 3 reaches the step budget, at 0.049.
On T20, whose solution is only `C^1`, the medians are 4.3e-3 (penalty MLP) and 1.4e-2 (ICNN) against a threshold of 1.3e-3.
On T1 and T8, which are smooth, `DirichletICNN` succeeds in every seed (on T8 eight seeds meet the gradient tolerance), and the penalty MLP in 9 (seed 1 ends at 1.79e-2, just above the threshold of 1.72e-2) and 5.

### Finding 5: `DirichletICNN` in 3D works in 7 seeds of 10, in 4D in none

On T9 (3D) eight seeds reach the step budget, seven of them below the threshold of 3.48e-2 (3.1e-3 to 2.9e-2) and seed 0 above it (9.2e-2); seed 4 stalls after 1843 steps at 0.83 and seed 9 after 3 steps at 1.9e3.
On T10 (4D) nine seeds stall, four of them after one step (errors 1.6e4 to 4.8e4), and the best seed ends at 2.5; `docs/methods/icnn.md` (Known limits) records first losses of 2e33 to 3e36 there, against 2e26 to 3e28 on T9.
The time median on T10 (155 s) is short because most seeds stop early, not because the solve is cheap.

### Finding 6: cost

At `n = 361` the classical presets take 2.6 s to 240 s on one thread, with 129600 unknowns on the square and 102328 on the disk; the neural presets take 416 s to 1450 s per seed (medians; T10's 155 s is the exception of Finding 5), and on T1, T2 and T20 their median errors are two to four orders of magnitude above the best classical preset's.
The cheapest accurate classical preset differs by problem: `FDPoisson` on T1 (2.6 s, 5.28e-7), `FDHybrid` on T2 (4.18e-5 in 152 s, against 5.41e-3 for `FDPoisson`), `FDPoisson` on T20 (1.59e-5 in 36 s).

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
