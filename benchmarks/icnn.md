# Input convex network: published versus ours

`DirichletICNN` with the preset's defaults (the paper's, where the paper gives them), seeds $0$ to $9$, against Nystrom and Vestberg (2023), arXiv 2110.03310v3, read from the PDF.
**Levels (ADR 0002): figure only on T1**, against Figure 2 (Example 4.2), which prints no table; **mismatch on T6**, against Table 1 (Example 4.6, the $\mathrm{stdev} = 0$ row).
The T6 mismatch holds under either reading of the equation term, which eq. (3.3) prints as a sum: read as a mean, the same network and optimizer meet the paper's stopping test and come about $3.6$ times closer, but stay about $3.6$ times above Table 1 (Finding 2).
`TransportICNN`, the input convex network of Caboussat and Peruso (2025) on the transport problem T12, is compared with their Table 2 at the end of this file: **statistical for the L-BFGS row's errors, mismatch for the Adam row's errors and for both rows' final loss** (T12, below); the L-BFGS errors agree although the same row's loss does not, and Finding 3 suggests that this agreement may be in part a coincidence.

Receipts: `icnn_t1.csv` and `icnn_t6.csv`, written by `run_icnn.py` on a100server at commit 08a29b7, CPU, one thread per seed (`THREADS_PER_SEED=1`, the `threads` column); every seed is an MLflow run named `seed=<s> preset` in `maesolvers-icnn-T1` and `maesolvers-icnn-T6`, with its training curve and the device and thread count as parameters.
The commits this file cites are on the branch of PR #55, which was squash-merged into `next` as 02a6ab6, so they are not in its history; GitHub keeps them under `refs/pull/55/head`.
The experiments also hold runs of the same names from earlier commits of this branch (Finding 1); the tag `mlflow.source.git.commit` tells them apart.
Bands are the median and the $5$% and $95$% quantiles over the ten seeds, computed by `maesolvers.metrics.summarize_errors` with ADR 0007's failed-seed rule; no seed failed.

## T1 (Example 4.2): figure only

Figure 2 plots the absolute error of the trained network on the unit square.
Read from the rendered page (p. 11): the colorbar runs from $0$ to $2.97 \times 10^{-5}$ (its top tick), the largest errors are on the boundary, near $(0.85, 1)$ and $(0, 0.7)$, and most of the square is below $10^{-5}$.
The paper reports one run, which "obtain[ed] convergence after $2287$ iterations" of SciPy's BFGS.

| Measure | Ours, median $[5, 95]$% |
|---|---|
| max, shared $201 \times 201$ grid (`evaluate`) | $5.14 \times 10^{-5}$ $[2.02 \times 10^{-5}, 1.93 \times 10^{-4}]$ |
| RMS, shared grid (`evaluate`) | $1.05 \times 10^{-5}$ $[3.99 \times 10^{-6}, 4.73 \times 10^{-5}]$ |
| max, $10^6$ random interior points (the paper's measure) | $5.13 \times 10^{-5}$ $[2.01 \times 10^{-5}, 1.93 \times 10^{-4}]$ |
| mean, $10^6$ random interior points (the paper's measure) | $7.19 \times 10^{-6}$ $[2.71 \times 10^{-6}, 3.54 \times 10^{-5}]$ |
| max on the boundary, $10^5$ random points | $5.14 \times 10^{-5}$ $[2.02 \times 10^{-5}, 1.93 \times 10^{-4}]$ |
| non-convex fraction, shared grid (`evaluate`) | $0$ in every seed |
| loss after $10000$ steps | $2.33 \times 10^{-7}$ $[9.91 \times 10^{-8}, 2.98 \times 10^{-6}]$ |

Verdict: the figure's largest error, $2.97 \times 10^{-5}$, lies inside our band of max errors, and our largest errors, like the figure's, are on the boundary (the boundary max equals the grid max in every seed).
That is agreement at the figure-only level; three seeds ($0$, $2$ and $8$) end at or below the figure's top, and the median is $1.7$ times above it.
One seed is clearly worse than the rest: seed $1$, max error $2.7 \times 10^{-4}$, with the highest final loss ($4.2 \times 10^{-6}$).
The network is convex at every point of the shared grid in every seed, as it must be by construction; the penalty MLP on the same problem leaves negative Hessian eigenvalues in $5$ of $10$ seeds (`penalty_mlp.md`, Finding 3).
Against the penalty MLP of the same paper on the same problem (`penalty_mlp.md`, max $7.06 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ on the shared grid), the ICNN is about $140$ times more accurate, as the paper says (errors "of the order $10^{-3}$" in Figure 1, "$`10^{-5}`$" in Figure 2).

## T6 (Example 4.6): mismatch

Table 1 gives the max and the average absolute error at $10^6$ random interior points of one run per noise level; the $\mathrm{stdev} = 0$ row is the run without noise, probably the run of Figure 7 (its colorbar ends near $8.5 \times 10^{-6}$; but Figure 7 is "the result after $5000$ iterations of BFGS", and the Table 1 runs went on to at most $10000$).
The paper trained the Table 1 runs "until convergence or at most $10{,}000$ iterations", with "the same network architecture as in Example 4.2", $3000$ interior and $400$ boundary points, the points on each piece of the boundary in proportion to its length.
Example 4.6 does not give the loss; we carry it over from Example 4.2 by assumption: $C = 100$, and the equation term also at the boundary points (which the paper's Section 5 leaves out).

| Measure | Table 1, $\mathrm{stdev} = 0$ (one run) | Ours, median $[5, 95]$% | Published inside the band |
|---|---|---|---|
| max error, $10^6$ random interior points | $8.58 \times 10^{-6}$ | $1.10 \times 10^{-4}$ $[5.49 \times 10^{-5}, 2.36 \times 10^{-4}]$ | no |
| average error, $10^6$ random interior points | $1.17 \times 10^{-6}$ | $1.49 \times 10^{-5}$ $[8.63 \times 10^{-6}, 3.71 \times 10^{-5}]$ | no |
| max, shared grid (`evaluate`) | | $9.88 \times 10^{-5}$ $[5.08 \times 10^{-5}, 2.27 \times 10^{-4}]$ | |
| RMS, shared grid (`evaluate`) | | $1.95 \times 10^{-5}$ $[1.15 \times 10^{-5}, 4.89 \times 10^{-5}]$ | |
| max on the boundary, $10^5$ random points | | $1.12 \times 10^{-4}$ $[5.55 \times 10^{-5}, 2.38 \times 10^{-4}]$ | |
| non-convex fraction, shared grid (`evaluate`) | | $0$ in every seed | |
| loss after the last step | | $1.18 \times 10^{-6}$ $[3.01 \times 10^{-7}, 2.92 \times 10^{-6}]$ | |

Verdict: mismatch (ADR 0002, level 5).
Both published numbers lie below our $5$% quantiles, the median errors are about $13$ times the published ones, and our best seed ($5$: max $4.3 \times 10^{-5}$, average $8.1 \times 10^{-6}$) is still $5$ and $7$ times above them.
Eight seeds stopped at the $10000$-step budget and two ($1$ and $2$) stalled on a failed line search after $8670$ and $9515$ steps, SciPy's "precision loss"; none met the stopping test (Finding 2), and the loss was still falling at step $10000$ in the eight (from $1.3 \times 10^{-6}$ to $1.2 \times 10^{-5}$ at step $5000$, the step of Figure 7, to $2.1 \times 10^{-7}$ to $3.2 \times 10^{-6}$ at step $10000$).
As on T1, the largest error is on the boundary: the boundary max is at least the interior max in every seed.
On T1 the figure's top lies inside our band, but near its lower end (three seeds of ten reach it); on T6 the table lies below it.
Read together, the paper's runs trained further than ours in fewer iterations, by about an order of magnitude in the error.
Finding 2 tests the one difference we found that could explain it, a mean in place of the printed sum in the equation term: with it, the same seeds meet the stopping test in $7$ of $10$, after a median of $4844$ steps, but Table 1 stays below the band (max $3.09 \times 10^{-5}$ $[1.21 \times 10^{-5}, 4.96 \times 10^{-5}]$, average $3.44 \times 10^{-6}$ $[1.67 \times 10^{-6}, 5.76 \times 10^{-6}]$), a mismatch about $3.6$ and $2.9$ times above it.
Only a $3400$ times larger BFGS start than SciPy's puts Table 1 inside the band.
The verdict here is the preset's, which follows the print, as ruled on #13; it is a mismatch under either reading.
The penalty MLP, run next to this table in `penalty_mlp.md`, ends at $6.10 \times 10^{-3}$ $[3.59 \times 10^{-3}, 1.02 \times 10^{-2}]$, $55$ times above the ICNN.

## Finding 1: the optimizer, not the start, decides the accuracy

The first receipts of this branch (MLflow runs `seed=<s> preset` tagged with commit f700c85, which added `run_icnn.py` to the network code of bc4c637) trained the network with `LBFGS()`, the optimizer of the penalty MLP: $100$ curvature pairs.
On T1 every seed stopped at the $10000$-step budget with a loss of $1.23 \times 10^{-2}$ $[1.87 \times 10^{-3}, 3.82 \times 10^{-2}]$ and a max error of $1.39 \times 10^{-2}$ $[5.22 \times 10^{-3}, 2.93 \times 10^{-2}]$ on the shared grid, about $500$ times the figure's top: a mismatch.
The loss was still falling, slowly; the paper's run had converged after $2287$ iterations.

Two settings differ from the paper's: the optimizer (the paper ran SciPy's BFGS, which keeps a dense inverse Hessian approximation) and the standard deviation of the initial parameters, which the paper does not give.
Diagnostic runs on T1 (MLflow `maesolvers-icnn-T1`, runs named `diagnostic ...` and `seed=<s> max_steps=3000 init_std=<std>`, commits 5529a71 and 3311534) separated the two:

| T1, $3000$ steps | max error, shared grid |
|---|---|
| `LBFGS()`, `init_std = 1`, seeds $0$, $2$, $6$ | not measured at step $3000$ (loss $0.07$ to $1.8$ there); $3.7 \times 10^{-3}$ to $2.9 \times 10^{-2}$ at step $10000$ |
| `LBFGS()`, `init_std = 0.3`, seeds $0$, $2$, $6$ | $1.37 \times 10^{-2}$ $[1.09 \times 10^{-2}, 2.88 \times 10^{-2}]$ |
| `LBFGS()`, `init_std = 0.05`, seeds $0$, $2$, $6$ | $7.3$ $[0.78, 10.6]$: two seeds stall, the line search fails |
| `LBFGS(history_size=10_000)`, seed $0$, `init_std = 1` / $0.3$ | $3.6 \times 10^{-2}$ / $6.5 \times 10^{-3}$ |
| SciPy's `minimize(method="BFGS")` on the same loss, seed $0$, `init_std = 1` / $0.3$ | $3.6 \times 10^{-4}$ / $7.2 \times 10^{-4}$ |

The start matters little and the optimizer a lot: SciPy's dense BFGS ends the same $3000$ iterations with errors $20$ to $100$ times smaller, from either start.
Longer memory alone does not close the gap; L-BFGS also differs from BFGS in rescaling its initial approximation at every step, which we did not test separately.
One effect of the start is visible, though: with `init_std = 1` the squared roots have mean $1$, the activations grow from layer to layer, the first loss is about $6 \times 10^{19}$, and L-BFGS shrinks $9$ of the $10$ output weights below $10^{-4}$ within $3000$ steps (`diagnostic dead roots ...`); a root at $0$ has a zero gradient ($dW/dV = 2V$), so these weights cannot grow back.

So the library got the optimizer part `BFGS`, which runs one iteration of SciPy's BFGS per training step, line search included, and the preset uses it; `init_std` stays $1$, the paper's "normal distribution" read literally.
Two intermediate versions of the part show how closely it has to follow SciPy (T1, seeds $0$ to $9$, $10000$ steps):

| `BFGS` version | max error, shared grid | seeds that stalled |
|---|---|---|
| torch's strong Wolfe line search (commit 3311534) | $1.47 \times 10^{-4}$ $[8.62 \times 10^{-5}, 3.08 \times 10^{-4}]$ | $5$ |
| SciPy's line search, update skipped when $y^\top s \le 10^{-10}$ (commit 4bd80b7) | $1.16 \times 10^{-4}$ $[6.19 \times 10^{-5}, 2.43 \times 10^{-4}]$ | $4$ |
| SciPy's line search and update (commit 08a29b7, the receipts) | $5.14 \times 10^{-5}$ $[2.02 \times 10^{-5}, 1.93 \times 10^{-4}]$ | $0$ |

Once the loss is near $10^{-6}$, $y^\top s$ falls below $10^{-10}$ and a skipped update starves the approximation; SciPy never skips.
SciPy's BFGS run directly on seeds $0$, $3$ and $6$ for $10000$ iterations (`diagnostic optimizer=bfgs ...`) gave $6.3 \times 10^{-5}$, $4.6 \times 10^{-5}$ and $6.9 \times 10^{-5}$, against $2.9 \times 10^{-5}$, $5.9 \times 10^{-5}$ and $4.9 \times 10^{-5}$ from the receipts: the same distribution; the runs differ in rounding, which $10000$ quasi-Newton steps amplify.
A test checks that `BFGS` reproduces SciPy's iterates on the Rosenbrock function to $10^{-9}$.

## Finding 2: a mean equation term meets the stopping test, but T6 stays a mismatch

The paper's T1 run stopped after $2287$ iterations on SciPy's test, a gradient of norm at most $10^{-5}$.
With the preset, no seed meets the test (SciPy's default form: largest gradient component at most $10^{-5}$) within $10000$ steps, on T1 or on T6; every T1 seed was still lowering its loss at step $10000$ (MLflow: the loss at step $2287$ is $3.3 \times 10^{-6}$ to $7.6 \times 10^{-5}$, at step $10000$ $9.1 \times 10^{-8}$ to $4.2 \times 10^{-6}$).
Eq. (3.3) prints the equation term as a sum over the $3400$ collocation points and the boundary term as a mean over the $400$ boundary points; the preset follows the print.
If the paper's code took the mean of the equation term instead, three facts follow.

1. **The same minimiser, but the BFGS start is not a detail.** The mean form, $(1/3400) \sum (\det D^2 u - f)^2 + 100 \operatorname{mean} (u - g)^2$, is $1/3400$ times the printed sum with `boundary_weight = 3.4e5`, so the two have the same minimiser.
   BFGS is invariant under a scaling of the loss if its start $H_0$ and its stopping test scale with it: SciPy's $H_0 = I$ and test $10^{-5}$ on the mean form are $H_0 = I/3400$ and test $0.034$ on the scaled sum.
   `run_icnn.py --mean-equation --boundary-weight 340000` runs the mean form itself, with $H_0 = I$ and the preset's `tol = 1e-5` (`MeanEquationLoss`, the preset's loss divided by the $3400$ points).
   `run_icnn.py --boundary-weight 340000 --tol 0.034` runs the scaled sum with $H_0 = I$, which is the mean form with a $3400$ times larger start, $H_0 = 3400 I$.
   That start decides the result: it moves the T6 max error from $3.1 \times 10^{-5}$ to $5.5 \times 10^{-6}$ (medians, table below).
2. **The reweighting acts on the boundary, where the max error is.** In every seed of all four runs below, the max on the boundary is at least the max inside.
   With the printed sum the root mean square boundary misfit is $4.1 \times 10^{-5}$ on T6 and $2.2 \times 10^{-5}$ on T1 (medians), the size of the max error.
   The mean reading lowers it to $1.1 \times 10^{-5}$ and $6.3 \times 10^{-6}$, the larger start to $2.3 \times 10^{-6}$ and $1.4 \times 10^{-6}$, and the max error falls with it.
   Their equation terms, as sums, end larger than with the printed sum (on T6, medians, $1.1 \times 10^{-4}$ and $7.3 \times 10^{-6}$ against $7.5 \times 10^{-7}$) because their runs stop earlier; with the stricter test the same minimiser ends below it ($1.7 \times 10^{-7}$), so the reweighting does not cost accuracy inside.
3. **The mean reading meets the stopping test, but on T6 it is still a mismatch; only the larger BFGS start puts Table 1 inside the band. Figure 2 does not tell the readings apart.**

| Seeds $0$ to $9$, median $[5, 95]$% | Paper | Printed sum (the preset, the receipts) | Mean reading (`--mean-equation`, `boundary_weight = 3.4e5`, `tol = 1e-5`, $H_0 = I$) | Mean reading with a $3400$ times larger BFGS start ($H_0 = 3400 I$: `boundary_weight = 3.4e5`, `tol = 0.034`) | Same minimiser, stricter test (`boundary_weight = 3.4e5`, `tol = 1e-5`) |
|---|---|---|---|---|---|
| T6 max error, $10^6$ random interior points | $8.58 \times 10^{-6}$ (Table 1) | $1.10 \times 10^{-4}$ $[5.49 \times 10^{-5}, 2.36 \times 10^{-4}]$ | $3.09 \times 10^{-5}$ $[1.21 \times 10^{-5}, 4.96 \times 10^{-5}]$ | $5.47 \times 10^{-6}$ $[2.58 \times 10^{-6}, 1.68 \times 10^{-5}]$ | $9.34 \times 10^{-7}$ $[5.19 \times 10^{-7}, 1.44 \times 10^{-5}]$ |
| T6 average error, $10^6$ random interior points | $1.17 \times 10^{-6}$ (Table 1) | $1.49 \times 10^{-5}$ $[8.63 \times 10^{-6}, 3.71 \times 10^{-5}]$ | $3.44 \times 10^{-6}$ $[1.67 \times 10^{-6}, 5.76 \times 10^{-6}]$ | $6.26 \times 10^{-7}$ $[2.87 \times 10^{-7}, 2.53 \times 10^{-6}]$ | $8.12 \times 10^{-8}$ $[4.53 \times 10^{-8}, 2.17 \times 10^{-6}]$ |
| T6 steps; stops (converged / budget / stalled) | at most $10000$ | $10000$; $0$ / $8$ / $2$ | $4844$; $7$ / $3$ / $0$ | $3033$; $8$ / $1$ / $1$ | $9762$; $0$ / $5$ / $5$ |
| T1 max error, shared grid | about $2.97 \times 10^{-5}$ (Figure 2, colorbar top) | $5.14 \times 10^{-5}$ $[2.02 \times 10^{-5}, 1.93 \times 10^{-4}]$ | $1.66 \times 10^{-5}$ $[8.44 \times 10^{-6}, 2.36 \times 10^{-5}]$ | $4.88 \times 10^{-6}$ $[2.08 \times 10^{-6}, 1.58 \times 10^{-5}]$ | $1.16 \times 10^{-6}$ $[5.56 \times 10^{-7}, 4.91 \times 10^{-6}]$ |
| T1 steps; stops (converged / budget / stalled) | $2287$, converged | $10000$; $0$ / $10$ / $0$ | $4114$; $8$ / $2$ / $0$ | $3863$; $7$ / $1$ / $2$ | $9451$; $0$ / $3$ / $7$ |

On T6 the mean reading leaves both published numbers below its $5$% quantiles: a mismatch (ADR 0002, level 5), the medians $3.6$ and $2.9$ times above Table 1.
That is about $3.6$ times closer than the printed sum ($13$ times above), but not the paper's accuracy: one seed of ten ($2$, max $9.4 \times 10^{-6}$, average $1.6 \times 10^{-6}$) comes near the table, and it is still above both numbers.
The mean reading does meet the stopping test, in $7$ of $10$ seeds on T6 (after $1834$ to $7444$ steps) and $8$ of $10$ on T1 (after $1677$ to $8462$ steps, the paper's $2287$ among them); with the printed sum no seed does.
On T1 all ten seeds end below the figure's top, the median $1.8$ times below it; with the printed sum the median is $1.7$ times above it, and the colorbar is too coarse a measure to choose between the two.
Only the $3400$ times larger start puts Table 1 inside the band on T6 (the medians $1.6$ and $1.9$ times below the table, $9$ of $10$ seeds below the published max), and that start is not SciPy's default $H_0 = I$ on either reading of eq. (3.3).
The stricter test (last column) trains on and ends about $10$ times below Table 1: the start and the stopping test, not only the weight, set where the run ends.
So neither reading with SciPy's defaults reproduces Table 1: the mean comes closer and stops as the paper's runs did, and a factor of about $3$ stays unexplained.
Under the mean reading with the larger start, too, a stalled seed is one whose line search failed; the paper restarted with Adam there (Example 4.1), which we do not.

The preset keeps the printed sum, and the T6 verdict above stays a mismatch; it would be a mismatch under the mean reading too.
Issue #13 ruled that `DirichletICNN` keeps the printed sum of eq. (3.3) as its default; the mean reading (a loss part with a mean equation term, as `MeanEquationLoss` in `run_icnn.py`) stays a benchmark diagnostic, as here.
These runs are diagnostics, not receipts, so their CSVs stay on a100server (`/jczhang02/maesolvers-data/logs/icnn-mean/`), CPU, one thread per seed:

- mean reading: MLflow runs `diagnostic seed=<s> boundary_weight=340000.0 mean_equation=True` in `maesolvers-icnn-T1` and `maesolvers-icnn-T6`, commit ef579fc (`mean-equation-t1.csv`, `mean-equation-t6.csv`); a review script, independent of `run_icnn.py`, gave the same T6 errors seed by seed (runs `review true-mean H0=I seed=<s>`);
- larger start and stricter test: runs `diagnostic mean-equation seed=<s> boundary_weight=340000.0 tol=0.034` and `... boundary_weight=340000.0`, commit 55c1469 (`icnn-mean-tol-t*.csv`, `icnn-mean-t*.csv`).

## Conventions that differ

- **Error measure.** The paper measures at $10^6$ random interior points (Section 4.6) or plots on the square (Figure 2); the library's primary error is on the shared evaluation grid (ADR 0007), and the receipts report both. Our $10^6$ points are uniform in the domain, drawn by rejection from one fixed generator seed ($2023$), the points of `run_penalty_mlp.py`.
- **Optimizer.** One iteration of SciPy's BFGS per training step (`BFGS`), up to rounding the paper's optimizer; the training stops on the largest gradient component (SciPy's default `norm=inf`), not on the Euclidean norm the paper's text names.
- **T6 loss.** Example 4.6 names only the architecture of Example 4.2; the boundary weight $C = 100$ and the equation term at the boundary points are carried over from Example 4.2 by assumption.
- **Budget.** $10000$ steps, the cap the paper states for Table 1; the paper gives no cap for Example 4.2, whose run converged after $2287$ iterations.
- **Initial parameters.** Normal with standard deviation $1$ for every parameter, the roots $V_j$ of the non-negative weights included; the paper says "the normal distribution" and no more.
- **Float64** throughout, as ADR 0008 requires; the paper does not say which precision it used (TensorFlow's default is float32).
- **Threads.** One torch thread per seed; the thread count is part of the setting, and `Result.info["threads"]` and the MLflow parameters record it.

## Not compared

- Tables 2 and 3 of Nystrom-Vestberg (T9 in 3D, T10 in 4D, with $4000$ interior and $1200$ or $2400$ boundary points, $20015$ and $17506$ BFGS iterations): `DirichletICNN` supports both problems, but these runs are not part of 0.1.
- Table 1 with noise in $f$ ($\mathrm{stdev} > 0$), and Tables 4 to 6 (average error against the number of collocation points, with the equation term at the interior points only, `equation_at_boundary=False`).
- Figures 3 to 6 (T3 and its variant with extra boundary points, the sphere of Example 4.4, T5).

## Rerun

```sh
just push
just remote-bg icnn-t1 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T1
# when it has finished:
just remote-bg icnn-t6 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T6
```

Each takes about $17$ minutes on a100server with ten CPU threads; run them one after the other, to stay within $16$ threads.

The diagnostics of Finding 2, for T6 (and likewise T1), each about $15$ to $30$ minutes, again one after the other:

```sh
D=/jczhang02/maesolvers-data/logs/icnn-mean
# mean reading
just remote-bg icnn-mean-t6 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T6 \
  --mean-equation --boundary-weight 340000 --name diagnostic --out $D/mean-equation-t6.csv
# 3400 times larger BFGS start; drop --tol 0.034 for the stricter test
just remote-bg icnn-start-t6 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_icnn.py T6 \
  --boundary-weight 340000 --tol 0.034 --name diagnostic --out $D/icnn-mean-tol-t6.csv
```

The second command names its runs `diagnostic seed=<s> ...`; the runs of Finding 2 from commit 55c1469 carry the prefix `diagnostic mean-equation`.

## T12 (Caboussat-Peruso 2025, Table 2): statistical for L-BFGS, mismatch for Adam and the final loss

`TransportICNN` with the preset's defaults, seeds $0$ to $9$, against A. Caboussat and A. Peruso, "Convex physics informed neural networks for the Monge-Ampere optimal transport problem", arXiv 2501.10162v2 (2025-08-16), read from the PDF (Zotero GV9DAREQ, attachment YG62XJ8G).
That is the arXiv version, not the journal version in Eng. Comput. 43(8), which we have not read; its numbers may differ.
Table 2 gives, for the disk-to-ellipse problem of Section 4.2 (T12), the average and the standard deviation over ten runs of the final loss and of the L2 error of $\nabla u$ on the training and on test points, for two schedules: Adam alone (Figure 5, "Adam only") and Adam for $200$ epochs then L-BFGS (Figure 5, "Adam - L-BFGS"), $2000$ epochs in all.
`run_icnn_transport.py` runs both, the second as `TransportICNN()` and the first as `TransportICNN(adam_steps=2000)`; the same seed gives both the same collocation points and the same initial network.

Receipt: `icnn_t12.csv`, written by `run_icnn_transport.py` on a100server, on one A100 (`device` cuda), ten seeds at a time, one CPU thread each.
Every seed is an MLflow run named `<schedule> seed=<s> preset` in `maesolvers-icnn-T12`, with its training curve.
The L-BFGS runs are tagged with commit 45598f1; the server checkout moved to 21970e5 while the Adam runs started, so they carry that tag.
The two commits differ only in `registry.py`, the tests and the guide, and the Adam runs of seeds $0$ to $2$ equal, to every printed digit, the diagnostic runs of the same seeds from commit af99748 (`diagnostic adam seed=<s> preset`).

| Table 2 | Paper, mean (std) over $10$ runs | Ours, mean (std) | Ours, median $[5, 95]$% over the seeds | Our mean's bootstrap $[5, 95]$% | Paper's mean inside: our seeds' band; our mean's band |
|---|---|---|---|---|---|
| Adam, final loss | $0.00150$ ($0.0043$) | $2.87 \times 10^{-4}$ ($1.42 \times 10^{-4}$) | $2.32 \times 10^{-4}$ $[1.83 \times 10^{-4}, 5.18 \times 10^{-4}]$ | $[2.23 \times 10^{-4}, 3.63 \times 10^{-4}]$ | no, above; no, above |
| Adam, L2 error (train) | $0.02399$ ($0.00327$) | $1.05 \times 10^{-2}$ ($7.33 \times 10^{-3}$) | $7.45 \times 10^{-3}$ $[3.71 \times 10^{-3}, 2.15 \times 10^{-2}]$ | $[7.04 \times 10^{-3}, 1.42 \times 10^{-2}]$ | no, above; no, above |
| Adam, L2 error (test) | $0.02504$ ($0.00389$) | $1.05 \times 10^{-2}$ ($7.28 \times 10^{-3}$) | $7.54 \times 10^{-3}$ $[3.74 \times 10^{-3}, 2.14 \times 10^{-2}]$ | $[7.08 \times 10^{-3}, 1.42 \times 10^{-2}]$ | no, above; no, above |
| L-BFGS, final loss | $1.13 \times 10^{-5}$ ($1.39 \times 10^{-5}$) | $1.40 \times 10^{-4}$ ($1.79 \times 10^{-5}$) | $1.38 \times 10^{-4}$ $[1.16 \times 10^{-4}, 1.66 \times 10^{-4}]$ | $[1.32 \times 10^{-4}, 1.49 \times 10^{-4}]$ | no, below; no, below |
| L-BFGS, L2 error (train) | $0.00267$ ($0.00376$) | $2.24 \times 10^{-3}$ ($9.41 \times 10^{-4}$) | $2.06 \times 10^{-3}$ $[1.45 \times 10^{-3}, 3.78 \times 10^{-3}]$ | $[1.83 \times 10^{-3}, 2.75 \times 10^{-3}]$ | yes; yes, $3$% below the top |
| L-BFGS, L2 error (test) | $0.00268$ ($0.00376$) | $2.24 \times 10^{-3}$ ($9.42 \times 10^{-4}$) | $2.05 \times 10^{-3}$ $[1.44 \times 10^{-3}, 3.78 \times 10^{-3}]$ | $[1.83 \times 10^{-3}, 2.75 \times 10^{-3}]$ | yes; yes, $3$% below the top |

`python benchmarks/run_icnn_transport.py --summary` prints every cell of ours from the receipt; the band of the mean is the $5$% and $95$% quantiles of the mean of ten seeds drawn with replacement, $10000$ resamples from numpy's generator seed $0$.

The L2 error is the root mean square of $\lvert \nabla u - \nabla u_{\mathrm{exact}} \rvert$ over the points (`gradient_rms` of `gradient_map_error`; Conventions, below), on the $800$ interior collocation points of the seed (train) and on the $31397$ points of the shared evaluation grid in the disk (test).
Verdict (ADR 0002): **statistical** for the errors of the L-BFGS row, the row the paper recommends, and for its errors only.
Table 2 prints a mean over ten runs, and ADR 0002 defines level 3 for a published single-run number inside our per-seed $5$% to $95$% band; it has no rule for a published mean.
We apply two checks, and the verdict rests on both.
The first is the letter of ADR 0002: the published mean, $2.67 \times 10^{-3}$, lies inside our per-seed band $[1.45 \times 10^{-3}, 3.78 \times 10^{-3}]$; the check is lenient for a mean, which varies much less than one run.
The second compares like with like, a mean with a mean: $2.67 \times 10^{-3}$ lies inside the bootstrap band of our ten-seed mean, $[1.83 \times 10^{-3}, 2.75 \times 10^{-3}]$, but only just, $3$% below its top; our mean, $2.24 \times 10^{-3}$, is $0.84$ times theirs.
Three facts weaken the agreement.
The spreads differ: the paper's standard deviation, $3.76 \times 10^{-3}$, is larger than its mean and four times ours, $9.4 \times 10^{-4}$.
The same row's final loss is a mismatch (below), and Finding 3 shows that a sampling under which our loss falls below the paper's brings our error to $3.0 \times 10^{-4}$, with the paper's mean three times above our $95$% quantile; if the paper's runs sampled that way, in some runs or all, the error agreement joins two different regimes and is in part a coincidence, not evidence that the method was reproduced.
The second check depends on the reading of the L2 error (Conventions): as the L2 norm over the disk, our mean's band is $[3.25 \times 10^{-3}, 4.88 \times 10^{-3}]$ and $2.67 \times 10^{-3}$ lies below it.
**Mismatch** for the Adam row: the published means lie above our per-seed $95$% quantiles ($2.40 \times 10^{-2}$ against $2.15 \times 10^{-2}$ on the training points) and far above the band of our mean ($[7.04 \times 10^{-3}, 1.42 \times 10^{-2}]$), and our Adam runs train further than the paper's (Finding 4).
**Mismatch** for the final loss of both rows: the paper's L-BFGS loss lies a factor $10$ below our band and below the level the exact map leaves at our independent boundary points (Finding 3), and its Adam loss a factor $3$ above our band.
No seed failed, and the network is convex at every point of the shared grid in every seed (`nonconvex_fraction` $0$), as it must be by construction.

Two more numbers that Table 2 does not print: the largest pointwise error of the map on the shared grid is $4.53 \times 10^{-3}$ $[2.93 \times 10^{-3}, 6.30 \times 10^{-3}]$ with L-BFGS and $1.53 \times 10^{-2}$ $[7.60 \times 10^{-3}, 3.37 \times 10^{-2}]$ with Adam, and `evaluate`'s max error of $u$, after both sides' means are subtracted, is $2.03 \times 10^{-3}$ $[1.34 \times 10^{-3}, 3.63 \times 10^{-3}]$ and $7.17 \times 10^{-3}$ $[3.28 \times 10^{-3}, 1.93 \times 10^{-2}]$.
Figure 4 of the paper, one L-BFGS run, shows pointwise errors of each component of $\nabla u$ up to $3.2 \times 10^{-4}$, a run better than any of ours under the sampling of the preset (our smallest largest error of the map is $2.2 \times 10^{-3}$), and close to our best under the sampling of Finding 3 ($3.8 \times 10^{-4}$ for the norm of the error).
The paper's standard deviations are large next to its means ($0.00376$ against $0.00267$ for L-BFGS): with ten positive errors, that says one or two runs far worse than the rest, and Figure 5's band ($5$% to $95$%, from about $2 \times 10^{-4}$ to $9 \times 10^{-3}$ for the L-BFGS error) says the same.
Our L-BFGS spread is narrower, $1.29 \times 10^{-3}$ to $4.64 \times 10^{-3}$ over the ten seeds.

### Finding 3: the paper's final L-BFGS loss lies ten times below the level the exact map leaves at independent boundary points

The transport term of eq. (3.4) compares two finite point sets: $\nabla u$ of the $800$ points on the circle and the $800$ points on the ellipse.
When the two are drawn independently, as `TransportSampler` draws them, the exact map does not make the term zero: each image lies some way from the nearest target point.
For the exact solution and the collocation points of seeds $0$ to $2$, the term is $1.11 \times 10^{-4}$, $1.67 \times 10^{-4}$ and $1.34 \times 10^{-4}$ (`into` $4.35 \times 10^{-5}$ plus `onto` $6.77 \times 10^{-5}$ for seed $0$), pinned in `tests/test_transport_icnn.py` (`test_the_exact_map_leaves_a_transport_term_at_independent_points`); `python benchmarks/run_icnn_transport.py --exact-map` prints it for seeds $0$ to $9$, $1.11 \times 10^{-4}$ to $1.73 \times 10^{-4}$, next to the trained network of each seed.
With the target points uniform in the parameter angle instead of in arc length (the same command, `parametric target`), it is $1.30 \times 10^{-4}$, $1.37 \times 10^{-4}$ and $1.31 \times 10^{-4}$.
This level is the loss of one map, the exact one, not a lower bound over maps: a network can move the images of the boundary points toward the drawn target points and lower the term below it.
Our trained networks end at the same level, $1.38 \times 10^{-4}$ $[1.16 \times 10^{-4}, 1.66 \times 10^{-4}]$, almost all of it in the transport term (`into` plus `onto`; the equation term is $1.2 \times 10^{-6}$ in the median), and within $4$% of the exact map's term at the same points: the final loss lies above it in seeds $0$, $2$, $4$, $6$ and $8$ (by $0.5$% to $3.1$%) and below it in seeds $1$, $3$, $5$, $7$ and $9$ (by $1.1$% to $3.3$%), and the transport term alone lies below it in eight seeds (seed $5$: $1.65 \times 10^{-4}$ against $1.73 \times 10^{-4}$, $4$% below).
So the $433$ parameters of the network bend the map only a little toward the drawn points, and its error of about $2 \times 10^{-3}$ is the error this network reaches under this sampling (seven times smaller under shared angles, below); how far a richer convex map could lower the term we did not measure.
The paper's mean final loss, $1.13 \times 10^{-5}$, is ten times below both the exact map's level and every trained seed of ours, and its best runs (Figure 5's band reaches $10^{-6}$) a hundred times.
The paper does not say how many target points it drew, but more target points do not lower the exact map's level much: they shrink `into`, while `onto`, a mean over the target points of the distance to the nearest of the $800$ images, stays where it is (seed $0$ with $8000$ target points: `into` $6.2 \times 10^{-7}$, `onto` $6.8 \times 10^{-5}$; `--exact-map` prints both halves with ten times the target points).
So, if the reported loss is eq. (3.4) with $C = 1$ and the means as printed, we see no way to reach it with independent uniform draws on the two curves and a network of this size, and the paper's boundary points probably were not such draws; this suggests, but does not prove, a different sampling or loss, and its code is not public and it does not describe how it reports the loss.

A sampling under which the exact map makes the term zero: the same random angles $t$ on both curves, $(\cos t, \sin t)$ on the circle and $(3.5 + 2 \cos t, 0.5 \sin t)$ on the ellipse, which the exact map $\nabla u(x) = (2 x_1 + 3.5, x_2 / 2)$ sends one onto the other.
That is a natural way to code the two boundaries, and the paper does not say how it drew them.
`run_icnn_transport.py --shared-angles` replaces the sampler by such a one, `SharedAngleSampler`, a subclass of `TransportSampler` in the script (MLflow runs `diagnostic <schedule> seed=<s> sampler=shared-angles`, `/jczhang02/maesolvers-data/logs/icnn-t12/shared-angles.csv`):

| Seeds $0$ to $9$, median $[5, 95]$% (mean) | Paper, mean | Independent points (the preset, the receipt) | Shared angles |
|---|---|---|---|
| L-BFGS, final loss | $1.13 \times 10^{-5}$ | $1.38 \times 10^{-4}$ $[1.16 \times 10^{-4}, 1.66 \times 10^{-4}]$ ($1.40 \times 10^{-4}$) | $1.08 \times 10^{-6}$ $[4.18 \times 10^{-7}, 6.89 \times 10^{-6}]$ ($2.53 \times 10^{-6}$) |
| L-BFGS, transport term (`into` + `onto`) | | $1.36 \times 10^{-4}$ $[1.14 \times 10^{-4}, 1.65 \times 10^{-4}]$ | $4.07 \times 10^{-7}$ $[1.26 \times 10^{-7}, 2.69 \times 10^{-6}]$ |
| L-BFGS, L2 error (train) | $2.67 \times 10^{-3}$ | $2.06 \times 10^{-3}$ $[1.45 \times 10^{-3}, 3.78 \times 10^{-3}]$ ($2.24 \times 10^{-3}$) | $3.03 \times 10^{-4}$ $[1.50 \times 10^{-4}, 8.10 \times 10^{-4}]$ ($4.26 \times 10^{-4}$) |
| Adam, final loss | $1.50 \times 10^{-3}$ | $2.32 \times 10^{-4}$ $[1.83 \times 10^{-4}, 5.18 \times 10^{-4}]$ ($2.87 \times 10^{-4}$) | $1.11 \times 10^{-4}$ $[6.41 \times 10^{-5}, 4.78 \times 10^{-4}]$ ($1.90 \times 10^{-4}$) |
| Adam, L2 error (train) | $2.40 \times 10^{-2}$ | $7.45 \times 10^{-3}$ $[3.71 \times 10^{-3}, 2.15 \times 10^{-2}]$ ($1.05 \times 10^{-2}$) | $5.20 \times 10^{-3}$ $[2.88 \times 10^{-3}, 1.92 \times 10^{-2}]$ ($8.71 \times 10^{-3}$) |

With shared angles the transport term of the trained network falls by a factor of about $300$, the L-BFGS loss by about $100$, to $1.1 \times 10^{-6}$, and the error to $3.0 \times 10^{-4}$, seven times below the preset's and at the level of the paper's Figure 4 and of the lower edge of its Figure 5 band.
Now the paper's means lie above our bands: its loss, $1.13 \times 10^{-5}$, above the $95$% quantile $6.9 \times 10^{-6}$, and its error, $2.67 \times 10^{-3}$, three times above the $95$% quantile $8.1 \times 10^{-4}$.
Neither sampling reproduces both columns of the L-BFGS row at once, so the paper's runs probably mixed the two regimes, or sampled in a third way; its large standard deviations fit a mixture.
The preset keeps independent draws: the shared angles use the exact map of this one problem (it preserves the parameter angle), which a solver cannot know, and independent points are what the paper's text describes ("two sets of boundary points").
`TransportSampler`'s docstring and `docs/methods/icnn.md` (Known limits) state the exact map's level.

### Finding 4: our Adam trains further than the paper's

After the $200$ Adam steps of the L-BFGS schedule, our loss is $7.5 \times 10^{-2}$ $[4.3 \times 10^{-2}, 1.8 \times 10^{-1}]$ (`loss_switch`), where Figure 5 shows about $0.2$ to $0.3$ at the switch, and after $2000$ Adam steps it is $2.3 \times 10^{-4}$ $[1.8 \times 10^{-4}, 5.2 \times 10^{-4}]$, where the paper's mean is $1.50 \times 10^{-3}$ and Figure 5's curve ends near $1.5 \times 10^{-3}$.
The errors follow: our medians are about a third of the paper's means, which lie just above our $95$% quantiles.
Two settings of the Adam phase are not in the paper: the learning rate (the preset takes torch's default, $10^{-3}$) and the initial network before the fit to the identity map (the preset draws every parameter from the standard normal distribution, as `DirichletICNN` does; the paper's network is a torch one, whose default initialisation is uniform with a width of $1/\sqrt{\text{fan-in}}$).
A diagnostic with `init_std = 0.3`, seeds $0$ to $2$ (`diagnostic <schedule> seed=<s> init_std=0.3`, `/jczhang02/maesolvers-data/logs/icnn-t12/d2-std03.csv`) makes Adam slower: error $1.49 \times 10^{-2}$ $[1.07 \times 10^{-2}, 1.59 \times 10^{-2}]$ on the training points, loss $3.6 \times 10^{-4}$; with L-BFGS it changes little ($1.91 \times 10^{-3}$, loss $1.3 \times 10^{-4}$).
We did not search for the setting that matches the figure: both are unstated, the defaults are torch's and the library's, and the L-BFGS row, the paper's recommendation, does not depend on them.

### Conventions that differ (T12)

- **L2 error.** The paper writes $\lVert \nabla u_{\mathrm{ex}} - \nabla u_{\mathrm{NN}} \rVert_{L^2(X)}$ and does not say how it computes it; we read it as the root mean square over the points, not as the L2 norm over the disk, which is $\sqrt{\pi} = 1.77$ times larger. The evidence is Figure 5: at epoch $0$, after the fit to the identity map, its error is about $4$, and the identity map's error is $3.54$ as a root mean square and $6.28$ as an L2 norm. Under the other reading our bands scale by $1.77$: against the per-seed bands the L-BFGS row stays inside (train $[2.56 \times 10^{-3}, 6.71 \times 10^{-3}]$) and the Adam row comes inside too ($[6.58 \times 10^{-3}, 3.82 \times 10^{-2}]$); against the bands of our mean the two rows swap, the L-BFGS mean falling below ($[3.25 \times 10^{-3}, 4.88 \times 10^{-3}]$) and the Adam mean coming inside ($[1.25 \times 10^{-2}, 2.52 \times 10^{-2}]$). So both verdicts depend on this reading, and Figure 5 is the evidence for ours.
- **Train and test points.** "Train" is the $800$ interior collocation points of the seed; the paper may include its boundary points. "Test" is our shared evaluation grid (ADR 0007); the paper's "validation set" is not described. The two columns agree within $4$% in every seed, ours as the paper's within $5$%.
- **The paper reports mean and standard deviation over ten runs**, which we print next to ours. ADR 0002 places a published single-run number against our per-seed $5$% to $95$% band and has no rule for a published mean; the verdict applies that rule to the mean and, as a second check, places the mean against the bootstrap band of our ten-seed mean (Verdict, above).
- **Epochs and steps.** One training step is one Adam step or one L-BFGS iteration (`torch.optim.LBFGS` with `max_iter=1`, strong Wolfe line search, step $1$, $100$ curvature pairs: Table 1). Table 1's "$`100`$ epochs of $20$ sub-iterations" are $2000$ iterations; Figure 5 counts an L-BFGS epoch as its $20$ sub-iterations and ends both schedules at $2000$, with the switch at $200$, so the preset takes $200$ Adam steps and $1800$ L-BFGS steps. The L-BFGS losses change by less than $1$% over the last $500$ steps, so the $200$ steps of difference between the two readings do not matter here.
- **Stopping.** Torch's L-BFGS stops a call of $20$ sub-iterations when the loss or the step changes by less than $10^{-9}$ (`tolerance_change`), and the next call goes on; our step budget does not stop early. The preset stops when the largest gradient component is at most $10^{-7}$, torch's `tolerance_grad`; no seed reached it (every run stopped at `max_steps`).
- **Initialisation.** The paper fits $\nabla u$ to the identity map before training and gives no details; `IdentityStart` fits it by L-BFGS, at most $1000$ iterations, at the interior and boundary collocation points. No fit met its tolerance; the largest misfit of a component after the fit is $1.8 \times 10^{-3}$ $[1.0 \times 10^{-3}, 3.4 \times 10^{-3}]$.
- **Network.** "$`4`$ hidden layers with $10$ neurons per layer", softplus, non-negative weights as squares: `ICNN(width=10, depth=4)`, $433$ parameters in 2D. Every input feeds every layer (the passthrough of the paper's eq. (3.1)).
- **Equation term.** The paper's residual is $\det D^2 u - f / g(\nabla u)$, with $f = g = 1 / \pi$; a `Transport` problem has the target density $1$ and $f = 1$ on T12, so the residual is $\det D^2 u - 1$ in both. It is a mean over the interior points (the paper's $E_{\mathrm{PDE}}$), the boundary weight is $C = 1$.
- **Target points.** $800$, as many as the boundary points; the paper gives $N_b = 800$ and does not count the target points separately. They are uniform in arc length on the ellipse (the `BoundarySampling` contract); Finding 3 shows that the choice moves the exact map's level by little.
- **Float64** throughout (ADR 0008); the paper does not say.
- **Device.** The receipt trained on an A100; `TransportICNN()` trains on the CPU by default, as the comparison table does, and rounds differently there.

### Rerun (T12)

```sh
just push
# pick a GPU with no process in `ssh a100server nvidia-smi`
just remote-bg icnn-t12 env CUDA_VISIBLE_DEVICES=2 python benchmarks/run_icnn_transport.py
# the table's cells from the receipt, and Finding 3's exact map's level; seconds, nothing trained
just remote python benchmarks/run_icnn_transport.py --summary
just remote python benchmarks/run_icnn_transport.py --exact-map
# Finding 3, shared angles; Finding 4, a smaller start (seeds 0 to 2)
D=/jczhang02/maesolvers-data/logs/icnn-t12
just remote-bg icnn-t12-shared10 env CUDA_VISIBLE_DEVICES=2 python benchmarks/run_icnn_transport.py \
  --shared-angles --workers 6 --name diagnostic --out $D/shared-angles.csv
just remote-bg icnn-t12-std env CUDA_VISIBLE_DEVICES=2 python benchmarks/run_icnn_transport.py \
  --seeds 0 1 2 --init-std 0.3 --workers 6 --name diagnostic --out $D/d2-std03.csv
```

The receipt takes about $17$ minutes: the ten L-BFGS runs about $14$ minutes each, side by side, the Adam runs under $2$ minutes.
