# Input convex network: published versus ours

`DirichletICNN` with the preset's defaults (the paper's, where the paper gives them), seeds 0 to 9, against Nystrom and Vestberg (2023), arXiv 2110.03310v3, read from the PDF.
**Levels (ADR 0002): figure only on T1**, against Figure 2 (Example 4.2), which prints no table; **mismatch on T6**, against Table 1 (Example 4.6, the `stdev = 0` row).
The T6 mismatch belongs to the equation term as eq. (3.3) prints it, a sum; read as a mean, the same network and optimizer reach the statistical level on T6 and meet the paper's stopping test (Finding 2).
The ICNN table of Caboussat and Peruso (2025) is not compared: it needs the transport boundary condition of 0.2 (Not compared, below).

Receipts: `icnn_t1.csv` and `icnn_t6.csv`, written by `run_icnn.py` on a100server at commit 08a29b7, CPU, one thread per seed (`THREADS_PER_SEED=1`, the `threads` column); every seed is an MLflow run named `seed=<s> preset` in `maesolvers-icnn-T1` and `maesolvers-icnn-T6`, with its training curve and the device and thread count as parameters.
The experiments also hold runs of the same names from earlier commits of this branch (Finding 1); the tag `mlflow.source.git.commit` tells them apart.
Bands are the median and the 5% and 95% quantiles over the ten seeds, computed by `maesolvers.metrics.summarize_errors` with ADR 0007's failed-seed rule; no seed failed.

## T1 (Example 4.2): figure only

Figure 2 plots the absolute error of the trained network on the unit square.
Read from the rendered page (p. 11): the colorbar runs from 0 to 2.97e-5 (its top tick), the largest errors are on the boundary, near `(0.85, 1)` and `(0, 0.7)`, and most of the square is below 1e-5.
The paper reports one run, which "obtain[ed] convergence after 2287 iterations" of SciPy's BFGS.

| Measure | Ours, median [5%, 95%] |
|---|---|
| max, shared 201 x 201 grid (`evaluate`) | 5.14e-5 [2.02e-5, 1.93e-4] |
| RMS, shared grid (`evaluate`) | 1.05e-5 [3.99e-6, 4.73e-5] |
| max, 10^6 random interior points (the paper's measure) | 5.13e-5 [2.01e-5, 1.93e-4] |
| mean, 10^6 random interior points (the paper's measure) | 7.19e-6 [2.71e-6, 3.54e-5] |
| max on the boundary, 10^5 random points | 5.14e-5 [2.02e-5, 1.93e-4] |
| non-convex fraction, shared grid (`evaluate`) | 0 in every seed |
| loss after 10000 steps | 2.33e-7 [9.91e-8, 2.98e-6] |

Verdict: the figure's largest error, 2.97e-5, lies inside our band of max errors, and our largest errors, like the figure's, are on the boundary (the boundary max equals the grid max in every seed).
That is agreement at the figure-only level; three seeds (0, 2 and 8) end at or below the figure's top, and the median is 1.7 times above it.
One seed is clearly worse than the rest: seed 1, max error 2.7e-4, with the highest final loss (4.2e-6).
The network is convex at every point of the shared grid in every seed, as it must be by construction; the penalty MLP on the same problem leaves negative Hessian eigenvalues in 5 of 10 seeds (`penalty_mlp.md`, Finding 3).
Against the penalty MLP of the same paper on the same problem (`penalty_mlp.md`, max 7.06e-3 [4.57e-3, 1.66e-2] on the shared grid), the ICNN is about 140 times more accurate, as the paper says (errors "of the order 1e-3" in Figure 1, "1e-5" in Figure 2).

## T6 (Example 4.6): mismatch

Table 1 gives the max and the average absolute error at 10^6 random interior points of one run per noise level; the `stdev = 0` row is the run without noise, probably the run of Figure 7 (its colorbar ends near 8.5e-6; but Figure 7 is "the result after 5000 iterations of BFGS", and the Table 1 runs went on to at most 10000).
The paper trained the Table 1 runs "until convergence or at most 10,000 iterations", with "the same network architecture as in Example 4.2", 3000 interior and 400 boundary points, the points on each piece of the boundary in proportion to its length.
Example 4.6 does not give the loss; we carry it over from Example 4.2 by assumption: `C = 100`, and the equation term also at the boundary points (which the paper's Section 5 leaves out).

| Measure | Table 1, `stdev = 0` (one run) | Ours, median [5%, 95%] | Published inside the band |
|---|---|---|---|
| max error, 10^6 random interior points | 8.58e-6 | 1.10e-4 [5.49e-5, 2.36e-4] | no |
| average error, 10^6 random interior points | 1.17e-6 | 1.49e-5 [8.63e-6, 3.71e-5] | no |
| max, shared grid (`evaluate`) | | 9.88e-5 [5.08e-5, 2.27e-4] | |
| RMS, shared grid (`evaluate`) | | 1.95e-5 [1.15e-5, 4.89e-5] | |
| max on the boundary, 10^5 random points | | 1.12e-4 [5.55e-5, 2.38e-4] | |
| non-convex fraction, shared grid (`evaluate`) | | 0 in every seed | |
| loss after the last step | | 1.18e-6 [3.01e-7, 2.92e-6] | |

Verdict: mismatch (ADR 0002, level 5).
Both published numbers lie below our 5% quantiles, the median errors are about 13 times the published ones, and our best seed (5: max 4.3e-5, average 8.1e-6) is still 5 and 7 times above them.
Eight seeds stopped at the 10000-step budget and two (1 and 2) stalled on a failed line search after 8670 and 9515 steps, SciPy's "precision loss"; none met the stopping test (Finding 2), and the loss was still falling at step 10000 in the eight (from 1.3e-6 to 1.2e-5 at step 5000, the step of Figure 7, to 2.1e-7 to 3.2e-6 at step 10000).
As on T1, the largest error is on the boundary: the boundary max is at least the interior max in every seed.
On T1 the figure's top lies inside our band, but near its lower end (three seeds of ten reach it); on T6 the table lies below it.
Read together, the paper's runs trained further than ours in fewer iterations, by about an order of magnitude in the error.
Finding 2 tests the one difference we found that could explain it, a mean in place of the printed sum in the equation term: with it, the same seeds put Table 1 inside the band (max 5.47e-6 [2.58e-6, 1.68e-5], average 6.26e-7 [2.87e-7, 2.53e-6]) and meet the stopping test in 8 of 10, after a median of 3033 steps.
The verdict here stays with the preset, which follows the print, until #13 decides between the two readings.
The penalty MLP, run next to this table in `penalty_mlp.md`, ends at 6.10e-3 [3.59e-3, 1.02e-2], 55 times above the ICNN.

## Finding 1: the optimizer, not the start, decides the accuracy

The first receipts of this branch (MLflow runs `seed=<s> preset` tagged with commit f700c85, which added `run_icnn.py` to the network code of bc4c637) trained the network with `LBFGS()`, the optimizer of the penalty MLP: 100 curvature pairs.
On T1 every seed stopped at the 10000-step budget with a loss of 1.23e-2 [1.87e-3, 3.82e-2] and a max error of 1.39e-2 [5.22e-3, 2.93e-2] on the shared grid, about 500 times the figure's top: a mismatch.
The loss was still falling, slowly; the paper's run had converged after 2287 iterations.

Two settings differ from the paper's: the optimizer (the paper ran SciPy's BFGS, which keeps a dense inverse Hessian approximation) and the standard deviation of the initial parameters, which the paper does not give.
Diagnostic runs on T1 (MLflow `maesolvers-icnn-T1`, runs named `diagnostic ...` and `seed=<s> max_steps=3000 init_std=<std>`, commits 5529a71 and 3311534) separated the two:

| T1, 3000 steps | max error, shared grid |
|---|---|
| `LBFGS()`, `init_std = 1`, seeds 0, 2, 6 | not measured at step 3000 (loss 0.07 to 1.8 there); 3.7e-3 to 2.9e-2 at step 10000 |
| `LBFGS()`, `init_std = 0.3`, seeds 0, 2, 6 | 1.37e-2 [1.09e-2, 2.88e-2] |
| `LBFGS()`, `init_std = 0.05`, seeds 0, 2, 6 | 7.3 [0.78, 10.6]: two seeds stall, the line search fails |
| `LBFGS(history_size=10_000)`, seed 0, `init_std = 1` / `0.3` | 3.6e-2 / 6.5e-3 |
| SciPy's `minimize(method="BFGS")` on the same loss, seed 0, `init_std = 1` / `0.3` | 3.6e-4 / 7.2e-4 |

The start matters little and the optimizer a lot: SciPy's dense BFGS ends the same 3000 iterations with errors 20 to 100 times smaller, from either start.
Longer memory alone does not close the gap; L-BFGS also differs from BFGS in rescaling its initial approximation at every step, which we did not test separately.
One effect of the start is visible, though: with `init_std = 1` the squared roots have mean 1, the activations grow from layer to layer, the first loss is about 6e19, and L-BFGS shrinks 9 of the 10 output weights below 1e-4 within 3000 steps (`diagnostic dead roots ...`); a root at 0 has a zero gradient (`dW/dV = 2V`), so these weights cannot grow back.

So the library got the optimizer part `BFGS`, which runs one iteration of SciPy's BFGS per training step, line search included, and the preset uses it; `init_std` stays 1, the paper's "normal distribution" read literally.
Two intermediate versions of the part show how closely it has to follow SciPy (T1, seeds 0 to 9, 10000 steps):

| `BFGS` version | max error, shared grid | seeds that stalled |
|---|---|---|
| torch's strong Wolfe line search (commit 3311534) | 1.47e-4 [8.62e-5, 3.08e-4] | 5 |
| SciPy's line search, update skipped when `y's <= 1e-10` (commit 4bd80b7) | 1.16e-4 [6.19e-5, 2.43e-4] | 4 |
| SciPy's line search and update (commit 08a29b7, the receipts) | 5.14e-5 [2.02e-5, 1.93e-4] | 0 |

Once the loss is near 1e-6, `y's` falls below 1e-10 and a skipped update starves the approximation; SciPy never skips.
SciPy's BFGS run directly on seeds 0, 3 and 6 for 10000 iterations (`diagnostic optimizer=bfgs ...`) gave 6.3e-5, 4.6e-5 and 6.9e-5, against 2.9e-5, 5.9e-5 and 4.9e-5 from the receipts: the same distribution; the runs differ in rounding, which 10000 quasi-Newton steps amplify.
A test checks that `BFGS` reproduces SciPy's iterates on the Rosenbrock function to 1e-9.

## Finding 2: a mean equation term closes the T6 gap and meets the stopping test

The paper's T1 run stopped after 2287 iterations on SciPy's test, a gradient of norm at most 1e-5.
With the preset, no seed meets the test (SciPy's default form: largest gradient component at most 1e-5) within 10000 steps, on T1 or on T6; every T1 seed was still lowering its loss at step 10000 (MLflow: the loss at step 2287 is 3.3e-6 to 7.6e-5, at step 10000 9.1e-8 to 4.2e-6).
Eq. (3.3) prints the equation term as a sum over the 3400 collocation points and the boundary term as a mean over the 400 boundary points; the preset follows the print.
If the paper's code took the mean of the equation term instead, three facts follow.

1. **The same minimiser, up to the scale.** The mean form, `(1/3400) sum (det D^2 u - f)^2 + 100 mean (u - g)^2`, is `1/3400` times the printed sum with `boundary_weight = 3.4e5`, so the two have the same minimiser.
   BFGS sees two differences: its start `H0 = I` is not invariant under a scaling of the loss, and its stopping test is: a largest gradient component of 1e-5 on the mean form is 0.034 on the scaled sum.
   So `run_icnn.py --boundary-weight 340000 --tol 0.034` runs the mean reading, up to `H0`; with the preset's `tol = 1e-5` it runs the same minimiser with a test 3400 times stricter.
2. **The reweighting acts on the boundary, where the max error is.** In every seed of all three runs below, the max on the boundary is at least the max inside; with the printed sum the root mean square boundary misfit is 4.1e-5 on T6 and 2.2e-5 on T1 (medians), the size of the max error.
   The mean reading lowers it to 2.3e-6 and 1.4e-6, and the max error falls with it; the equation term, as a sum, is larger at its end than with the printed sum (7.3e-6 against 7.5e-7 on T6, medians): the reading trades a larger residual inside for a smaller misfit on the boundary.
3. **The mean reading reaches the statistical level against Table 1; Figure 2 does not tell the two readings apart.**

| Seeds 0 to 9, median [5%, 95%] | Paper | Printed sum (the preset, the receipts) | Mean reading (`boundary_weight = 3.4e5`, `tol = 0.034`) | Same minimiser, stricter test (`boundary_weight = 3.4e5`, `tol = 1e-5`) |
|---|---|---|---|---|
| T6 max error, 10^6 random interior points | 8.58e-6 (Table 1) | 1.10e-4 [5.49e-5, 2.36e-4] | 5.47e-6 [2.58e-6, 1.68e-5] | 9.34e-7 [5.19e-7, 1.44e-5] |
| T6 average error, 10^6 random interior points | 1.17e-6 (Table 1) | 1.49e-5 [8.63e-6, 3.71e-5] | 6.26e-7 [2.87e-7, 2.53e-6] | 8.12e-8 [4.53e-8, 2.17e-6] |
| T6 steps; stops (converged / budget / stalled) | at most 10000 | 10000; 0 / 8 / 2 | 3033; 8 / 1 / 1 | 9762; 0 / 5 / 5 |
| T1 max error, shared grid | about 2.97e-5 (Figure 2, colorbar top) | 5.14e-5 [2.02e-5, 1.93e-4] | 4.88e-6 [2.08e-6, 1.58e-5] | 1.16e-6 [5.56e-7, 4.91e-6] |
| T1 steps; stops (converged / budget / stalled) | 2287, converged | 10000; 0 / 10 / 0 | 3863; 7 / 1 / 2 | 9451; 0 / 3 / 7 |

On T6 the mean reading puts both published numbers inside its band, the level "statistical" of ADR 0002; the median errors are 1.6 and 1.9 times below the table, and 9 of 10 seeds are below the published max (seed 1, stalled after 8412 steps, is the exception, at 2.4e-5).
It also meets the stopping test, in 8 of 10 seeds on T6 and 7 of 10 on T1, after a few thousand steps (T1: 696 to 7906 steps for the seeds that converged, the paper's 2287 among them).
On T1 all ten seeds end below the figure's top, the median about 6 times below it; with the printed sum the median is 1.7 times above it.
A figure's colorbar is too coarse a measure to choose between the two; Table 1, the stopping test and the step count all side with the mean.
The stricter test (last column) trains on and ends about 10 times below Table 1: the stopping test, not only the weight, sets where the run ends.
Under the mean reading, too, a stalled seed is one whose line search failed; the paper restarted with Adam there (Example 4.1), which we do not.

The preset keeps the printed sum, and the T6 verdict above stays a mismatch: the receipts are the preset's.
Whether `DirichletICNN` should follow the mean reading (`boundary_weight = 3.4e5`, `tol = 0.034`, or a loss part with a mean equation term) is asked on #13.
These runs are the MLflow runs named `diagnostic mean-equation seed=<s> boundary_weight=340000.0` (and `... tol=0.034`) in `maesolvers-icnn-T1` and `maesolvers-icnn-T6`, at commit 55c1469, CPU, one thread per seed; they are diagnostics, not receipts, so their CSVs stay on a100server (`/jczhang02/maesolvers-data/logs/icnn-mean/`).

## Conventions that differ

- **Error measure.** The paper measures at 10^6 random interior points (Section 4.6) or plots on the square (Figure 2); the library's primary error is on the shared evaluation grid (ADR 0007), and the receipts report both. Our 10^6 points are uniform in the domain, drawn by rejection from one fixed generator seed (2023), the points of `run_penalty_mlp.py`.
- **Optimizer.** One iteration of SciPy's BFGS per training step (`BFGS`), up to rounding the paper's optimizer; the training stops on the largest gradient component (SciPy's default `norm=inf`), not on the Euclidean norm the paper's text names.
- **T6 loss.** Example 4.6 names only the architecture of Example 4.2; the boundary weight `C = 100` and the equation term at the boundary points are carried over from Example 4.2 by assumption.
- **Budget.** 10000 steps, the cap the paper states for Table 1; the paper gives no cap for Example 4.2, whose run converged after 2287 iterations.
- **Initial parameters.** Normal with standard deviation 1 for every parameter, the roots `V_j` of the non-negative weights included; the paper says "the normal distribution" and no more.
- **Float64** throughout, as ADR 0008 requires; the paper does not say which precision it used (TensorFlow's default is float32).
- **Threads.** One torch thread per seed; the thread count is part of the setting, and `Result.info["threads"]` and the MLflow parameters record it.

## Not compared

- Caboussat and Peruso (2025), Table 2: an ICNN on T12, the transport problem from the unit disk onto an ellipse, with the L2 error of `grad u` over 10 runs. It needs the transport boundary condition, which comes in 0.2; the reproduction is tracked in #29. `DirichletICNN` does not support T12.
- Tables 2 and 3 of Nystrom-Vestberg (T9 in 3D, T10 in 4D, with 4000 interior and 1200 or 2400 boundary points, 20015 and 17506 BFGS iterations): `DirichletICNN` supports both problems, but these runs are not part of 0.1.
- Table 1 with noise in `f` (`stdev > 0`), and Tables 4 to 6 (average error against the number of collocation points, with the equation term at the interior points only, `equation_at_boundary=False`).
- Figures 3 to 6 (T3 and its variant with extra boundary points, the sphere of Example 4.4, T5).

## Rerun

```sh
just push
just remote-bg icnn-t1 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T1
# when it has finished:
just remote-bg icnn-t6 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T6
```

Each takes about 17 minutes on a100server with ten CPU threads; run them one after the other, to stay within 16 threads.

The mean reading of Finding 2, the same way, for T1 and T6 (drop `--tol 0.034` for its last column):

```sh
just remote-bg icnn-mean-t6 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T6 \
  --boundary-weight 340000 --tol 0.034 --name "diagnostic mean-equation" --out /tmp/icnn-mean-tol-t6.csv
```
