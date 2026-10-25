# Penalty MLP: published versus ours

`PenaltyMLP` with the preset's defaults (the paper's, where the paper gives them), seeds $0$ to $9$, against Nystrom and Vestberg (2023), arXiv 2110.03310v3, read from the PDF.
**Level (ADR 0002): figure only**, on T1 against Figure 1: the paper prints no table for this method (Finding 1).
On T6 we ran the method next to Table 1, which belongs to the paper's other method; that comparison is context, not a reproduction.

Receipts: `penalty_mlp_t1.csv` and `penalty_mlp_t6.csv`, written by `run_penalty_mlp.py` on a100server at commit da3df82, CPU, one thread per seed (`THREADS_PER_SEED=1`, the `threads` column); every seed is an MLflow run named `seed=<s> preset` in `maesolvers-penalty-mlp-T1` and `maesolvers-penalty-mlp-T6`, with its training curve and the device and thread count as parameters.
The commits this file cites are on the branch of PR #50, which was squash-merged into `next` as c15c545, so they are not in its history; GitHub keeps them under `refs/pull/50/head`; commit 6adc9a1 (Finding 4) is on the branch of PR #59.
The receipts reproduce only with one thread per seed: the rounding of the fits of $G$ and $D$, and so every later number, depends on the thread count.
Bands are the median and the $5$% and $95$% quantiles over the ten seeds, computed by `maesolvers.metrics.summarize_errors` with ADR 0007's failed-seed rule; no seed failed, every seed stopped at the $5000$-step budget (`stop = max_steps`), and in every seed both fits of the ansatz met their tolerance (`fits_converged`).

The receipts of review round 1 replace those of commit b4596fc, whose fit of $D$ stopped at a cap of $1000$ L-BFGS iterations in $19$ of $20$ seeds, before the tolerance the paper's fit runs to.
`HardDirichlet.fit_steps` is now $10000$, and $D$ converges in $852$ to $2752$ iterations; $G$ converged before and is unchanged, and so are the points and the initial $N$.
The bands moved because the fit of $D$ now follows the paper, not because of tuning: T1 from $7.59 \times 10^{-3}$ $[4.69 \times 10^{-3}, 2.22 \times 10^{-2}]$ to $7.06 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ (shared-grid max), T6 from $5.31 \times 10^{-3}$ $[3.18 \times 10^{-3}, 9.25 \times 10^{-3}]$ to $5.45 \times 10^{-3}$ $[3.46 \times 10^{-3}, 9.42 \times 10^{-3}]$.

## Finding 1: the paper's tables belong to the ICNN, not to the penalty MLP

The paper has two methods: the penalty MLP of Section 3.1 and an input convex network (ICNN) of Section 3.2.
The penalty MLP is used once, in Example 4.1 on T1, and its only result is Figure 1.
Every later example uses the ICNN:

- Section 4, before Example 4.1: the first example uses the penalty method, "in the subsequent ... examples" the input convex networks.
- Example 4.6 (T6, Table 1): "We use the same network architecture as in Example 4.2", and Example 4.2 is the ICNN ($5 \times 10$ softplus, $563$ parameters, loss $E_e + 100 E_b$ with the boundary points also in $E_e$, BFGS).
  Figure 7, the same example without noise after $5000$ BFGS iterations, has a colorbar that ends at about $8.5 \times 10^{-6}$, the $\mathrm{stdev} = 0$ row of Table 1.
- Section 4.7 (Tables 2 and 3, T9 and T10): "we use input convex networks to solve higher-dimensional equivalents"; Section 3.1 says the penalty "essentially only works in the two-dimensional case".

The research survey's plan (reproduce the penalty MLP against Table 1 and the 3D tables) is therefore wrong, and issue #12 inherited it.
The finding is on #12 and #13: the ICNN of #13 is compared against Table 1 on T6 in `icnn.md` (a mismatch), and can be against Tables 2 and 3 once it runs in 3D and 4D.
`papers.md` is corrected.

## T1 (Example 4.1): figure only

Figure 1 plots the absolute error of the trained network on the unit square.
Read from the rendered page (p. 10): the colorbar runs from $0$ to about $5.7 \times 10^{-3}$ (its top tick is $0.0054$), most of the square is below $1.5 \times 10^{-3}$, and the largest errors are at the boundary, near $(0.1, 0)$, $(1, 0.9)$ and the corner $(0, 1)$.
The paper gives no iteration count for this run, and restarted BFGS with Adam when the line search failed.

| Measure | Ours, median $[5, 95]$% |
|---|---|
| max, shared $201 \times 201$ grid (`evaluate`) | $7.06 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ |
| RMS, shared grid (`evaluate`) | $1.09 \times 10^{-3}$ $[6.32 \times 10^{-4}, 3.77 \times 10^{-3}]$ |
| max, $10^6$ random interior points (the paper's measure) | $6.88 \times 10^{-3}$ $[4.42 \times 10^{-3}, 1.65 \times 10^{-2}]$ |
| mean, $10^6$ random interior points (the paper's measure) | $8.26 \times 10^{-4}$ $[4.76 \times 10^{-4}, 2.98 \times 10^{-3}]$ |
| max on the boundary, $10^5$ random points | $7.05 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ |

Verdict: the figure's largest error, about $5.7 \times 10^{-3}$, lies inside our band of max errors, and our errors, like the figure's, are largest at the boundary.
That is agreement at the figure-only level; the figure gives one run and a colour scale, so no closer statement is possible.
Two seeds ($1$ and $8$) are clearly worse than the rest, with max errors of $1.8 \times 10^{-2}$ and $1.5 \times 10^{-2}$: their loss is the highest after $5000$ steps ($3.1 \times 10^{3}$ and $5.5 \times 10^{2}$), and their boundary error is mostly the leak $D N$ of an $N$ that trained badly (Finding 2).

## T6 (Example 4.6): our runs next to the ICNN's Table 1

| Measure | Table 1, $\mathrm{stdev} = 0$ (ICNN, one run) | PenaltyMLP, median $[5, 95]$% |
|---|---|---|
| max error, $10^6$ random interior points | $8.58 \times 10^{-6}$ | $6.10 \times 10^{-3}$ $[3.59 \times 10^{-3}, 1.02 \times 10^{-2}]$ |
| average error, $10^6$ random interior points | $1.17 \times 10^{-6}$ | $6.60 \times 10^{-4}$ $[3.81 \times 10^{-4}, 1.49 \times 10^{-3}]$ |
| max, shared grid (`evaluate`) | | $5.45 \times 10^{-3}$ $[3.46 \times 10^{-3}, 9.42 \times 10^{-3}]$ |
| RMS, shared grid (`evaluate`) | | $9.20 \times 10^{-4}$ $[5.60 \times 10^{-4}, 1.99 \times 10^{-3}]$ |
| max on the boundary, $10^5$ random points | | $6.15 \times 10^{-3}$ $[3.59 \times 10^{-3}, 1.03 \times 10^{-2}]$ |

The penalty MLP is about $700$ times less accurate than the paper's ICNN on T6, in line with the paper's own comparison on T1 (errors of order $10^{-3}$ for the penalty MLP in Figure 1, $10^{-5}$ for the ICNN in Figure 2).
This is a comparison of two methods, with no reproduction verdict.

## Finding 2: the error sits on the boundary, as the misfit of $G$ plus the leak $D N$

In every seed of both problems the max error at the $10^5$ boundary points is at least the max error at the $10^6$ interior points, and at most $4$% above it (the CSVs per seed): the error is largest on the boundary, and the interior points closest to it come near.
The interior error is smaller: the mean error is a tenth to a fifth of the max (tables above).

On the boundary the ansatz gives $u - g = (G - g) + D N$, so the boundary error has two sources: the misfit of $G$, the network of $20$ tanh units fitted to $g$, and the leak of $N$ through $D$, which does not vanish on the boundary.
The receipts split it at the same $10^5$ points (`lift_misfit_max` = max $\lvert G - g \rvert$, `distance_boundary_max` = max $\lvert D \rvert$, `leak_max` = max $\lvert D N \rvert$; the maxima lie at different points, so they do not add up exactly):

| Problem | max $\lvert G - g \rvert$ | max $\lvert D \rvert$ | max $\lvert D N \rvert$ | seeds where the leak is the larger term |
|---|---|---|---|---|
| T1 | $6.10 \times 10^{-3}$ $[3.89 \times 10^{-3}, 1.15 \times 10^{-2}]$ | $1.69 \times 10^{-2}$ $[1.28 \times 10^{-2}, 2.36 \times 10^{-2}]$ | $3.05 \times 10^{-3}$ $[1.26 \times 10^{-3}, 1.35 \times 10^{-2}]$ | $1$, $4$, $8$ |
| T6 | $4.18 \times 10^{-3}$ $[2.96 \times 10^{-3}, 7.43 \times 10^{-3}]$ | $1.56 \times 10^{-2}$ $[9.97 \times 10^{-3}, 2.55 \times 10^{-2}]$ | $3.81 \times 10^{-3}$ $[2.23 \times 10^{-3}, 7.45 \times 10^{-3}]$ | $1$, $5$, $7$, $8$, $9$ |

Both fits meet the paper's tolerance in every seed, yet $G$ misses $g$ by up to about $10^{-2}$ and $D$ is about $10^{-2}$ to $2.5 \times 10^{-2}$ on the boundary: the fits stop on the gradient of a mean square over a few hundred points, which bounds the average misfit, not the misfit between the points.
$D N$ then grows with $\lvert N \rvert$, so the boundary error depends on how $N$ trains.
The two worst T1 seeds show it: seed $1$ ($\lvert G - g \rvert$ $6.6 \times 10^{-3}$, $\lvert D N \rvert$ $1.45 \times 10^{-2}$) and seed $8$ ($4.0 \times 10^{-3}$, $1.23 \times 10^{-2}$) are worst because of the leak of a badly trained $N$, and seed $8$ has one of the smallest $G$ misfits of the ten.
In seed $0$, by contrast, $\lvert G - g \rvert$ ($1.29 \times 10^{-2}$) is the larger term, and $D N$ partly cancels it.
`Result.info["network"]` records the same split at the $400$ boundary collocation points (`lift_boundary_misfit`, `distance_boundary_max`), with the iterations and final gradient of each fit.

How much the fits matter shows when only their tolerance changes.
`run_penalty_mlp.py T1 --fit-tol 1e-7` (MLflow `maesolvers-penalty-mlp-T1`, runs `seed=<s> fit_tol=1e-07`, commit da3df82, CPU, one thread per seed; CSV on a100server at `/jczhang02/maesolvers-data/data/penalty_mlp_t1_fit_tol.csv`, not in the repository; command under Rerun) keeps every other setting, the cap of $10000$ iterations included; no fit meets $10^{-7}$ within it, so every fit runs the full $10000$ iterations:

| T1, seeds $0$ to $9$ | `fit_tol = 1e-5` (preset) | `fit_tol = 1e-7` |
|---|---|---|
| max, shared grid | $7.06 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ | $2.83 \times 10^{-3}$ $[1.22 \times 10^{-3}, 7.20 \times 10^{-3}]$ |
| max on the boundary | $7.05 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ | $1.84 \times 10^{-3}$ $[9.69 \times 10^{-4}, 3.55 \times 10^{-3}]$ |
| max $\lvert G - g \rvert$ on the boundary | $6.10 \times 10^{-3}$ $[3.89 \times 10^{-3}, 1.15 \times 10^{-2}]$ | $1.81 \times 10^{-3}$ $[1.35 \times 10^{-3}, 2.41 \times 10^{-3}]$ |
| max $\lvert D \rvert$ on the boundary | $1.69 \times 10^{-2}$ $[1.28 \times 10^{-2}, 2.36 \times 10^{-2}]$ | $1.17 \times 10^{-2}$ $[8.42 \times 10^{-3}, 1.54 \times 10^{-2}]$ |
| max $\lvert D N \rvert$ on the boundary | $3.05 \times 10^{-3}$ $[1.26 \times 10^{-3}, 1.35 \times 10^{-2}]$ | $1.04 \times 10^{-3}$ $[2.77 \times 10^{-4}, 3.71 \times 10^{-3}]$ |

The tighter fits cut the boundary error about fourfold, and in $7$ of $10$ seeds the max error moves off the boundary into the interior.
The paper's tolerance is therefore what makes the boundary dominate; the preset keeps it, since it is the paper's setting, and Figure 1's $5.7 \times 10^{-3}$ lies inside the band either way, so the level stays figure only.

## Finding 3: the penalty does not keep the network convex everywhere

The penalty acts at the collocation points only.
On T1, `evaluate` finds a negative Hessian eigenvalue at some points of the shared grid in $5$ of $10$ seeds, at up to $6.7$% of the points (seed $1$, the seed that trained worst), and at most $0.91$% in the others; on T6 in $2$ seeds ($0.19$% and $0.003$%).
The FD solvers report `nonconvex_fraction` $0$ on T1.

## Finding 4: dense BFGS lowers the loss but not the max error

`DirichletICNN` reaches its accuracy only with SciPy's dense BFGS (`icnn.md`, Finding 1), so the penalty MLP was also trained with `BFGS()` in place of `LBFGS()`, every other setting the preset's, the fits of $G$ and $D$ included.
`run_penalty_mlp.py T1 --optimizer bfgs` (MLflow `maesolvers-penalty-mlp-T1`, runs `seed=<s> optimizer=bfgs`, commit 6adc9a1, CPU, one thread per seed; CSV on a100server at `/jczhang02/maesolvers-data/data/penalty_mlp_t1_bfgs.csv`, not in the repository; command under Rerun):

| T1, seeds $0$ to $9$ | `LBFGS()` (preset) | `BFGS()` |
|---|---|---|
| max, shared grid | $7.06 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ | $7.05 \times 10^{-3}$ $[4.58 \times 10^{-3}, 3.39 \times 10^{-1}]$ |
| mean, $10^6$ random interior points | $8.26 \times 10^{-4}$ $[4.76 \times 10^{-4}, 2.98 \times 10^{-3}]$ | $5.42 \times 10^{-4}$ $[2.91 \times 10^{-4}, 3.14 \times 10^{-3}]$ |
| max $\lvert D N \rvert$ on the boundary | $3.05 \times 10^{-3}$ $[1.26 \times 10^{-3}, 1.35 \times 10^{-2}]$ | $3.03 \times 10^{-3}$ $[1.29 \times 10^{-3}, 3.31 \times 10^{-1}]$ |
| how the training stopped | $10$ at `max_steps` | $7$ at `max_steps`, $3$ `stalled` (seeds $1$, $5$, $8$) |

- In the eight seeds other than $1$ and $8$, BFGS ends with a loss $40$ to $15000$ times lower and a mean interior error $8$% to $54$% lower, but the max error agrees with L-BFGS's to within $1.2$%: it sits on the boundary, where it is the misfit of $G$ and the leak $D N$ (Finding 2), and the fits of $G$ and $D$ do not depend on the optimizer of $N$.
- Seeds $1$ and $8$ stop early on a failed line search and end worse: max errors $2.55 \times 10^{-2}$ and $5.95 \times 10^{-1}$ against $1.79 \times 10^{-2}$ and $1.50 \times 10^{-2}$, both from the leak $D N$ ($2.62 \times 10^{-2}$ and $5.80 \times 10^{-1}$).
  The paper, when the line search fails, trains a few hundred epochs with Adam at a learning rate of the order $10^{-7}$ and then goes on with BFGS (p. 10), which we do not do, so these two seeds, and the top of the BFGS band, show the stall, not the paper's procedure.

So dense BFGS does not change the max error, which the fits of $G$ and $D$ set (Finding 2); Figure 1's $5.7 \times 10^{-3}$ stays inside the band, the preset keeps `LBFGS()`, and the level stays figure only.

## Conventions that differ

- **Error measure.** The paper measures at $10^6$ random interior points (Section 4.6); the library's primary error is on the shared evaluation grid (ADR 0007), and the receipts report both.
  Our $10^6$ points are uniform in the domain, drawn by rejection from one fixed generator seed ($2023$) for every run.
  The two max errors differ by at most $4$% on T1, whose shared grid includes the boundary of the box, and the grid's is up to $17$% lower on T6, whose shared grid keeps interior points only: the max error lies on the boundary, which random interior points approach more closely than the grid's interior points.
- **Optimizer.** L-BFGS with $100$ curvature pairs and a strong Wolfe line search, one iteration per step, for SciPy's full BFGS; no Adam restarts on a failed line search. The fits of $G$ and $D$ use the same L-BFGS.
- **Stopping test of the fits.** The paper's text (Section 4) says SciPy's BFGS stops when the Euclidean norm of the gradient is below $10^{-5}$; SciPy's default test, and ours, is the largest component (`norm=inf`), looser by up to the square root of the number of parameters ($9$ for $G$, $13$ for $D$). We follow SciPy, which the paper ran.
- **Cap of the fits.** The paper caps neither fit; SciPy's own default cap is $200$ iterations per parameter ($16200$ for $G$, $32200$ for $D$). `fit_steps = 10000` is not reached on T1 or T6 (at most $557$ iterations for $G$, $2752$ for $D$); the receipts record the iterations of each fit.
- **Budget.** $5000$ L-BFGS steps; the paper gives no count for Example 4.1 ($5000$ BFGS iterations is the budget of its Example 4.6).
  No seed met the gradient tolerance within it; the loss was still decreasing.
- **Initial weights.** Standard normal, as the paper says; it does not give the standard deviation, and $1$ is TensorFlow's default.
- **Distance network $D$.** Fitted to the exact distance on both problems; the paper does so on the unit square and points to Berg and Nystrom (2018) for other domains.
- **Float64** throughout, as ADR 0008 requires; the paper does not say which precision it used.
- **Threads.** One torch thread per seed. With four threads, T1 seed $0$ fits a different $G$ (a reviewer measured max $\lvert G - g \rvert$ $7.4 \times 10^{-3}$ against $1.29 \times 10^{-2}$ with one thread), so the thread count is part of the setting; `Result.info["threads"]` and the MLflow parameters record it.

## Device

The receipts train on the CPU.
The networks are small, so a training step is bound by the launch of many small kernels: on a100server one A100 took about $70$ ms per step for one process and about $200$ ms per step for each of four processes sharing it, while one CPU thread took about $70$ ms per step and ten seeds ran side by side in about $8$ minutes.
Seeds $0$ to $4$ of T6 also ran on an A100 at the earlier commit 6bb1f9e, with the fit of $D$ capped at $1000$ iterations (MLflow): their shared-grid max errors were $4.810 \times 10^{-3}$, $6.792 \times 10^{-3}$, $5.256 \times 10^{-3}$, $1.124 \times 10^{-2}$ and $4.991 \times 10^{-3}$, against $4.809 \times 10^{-3}$, $6.795 \times 10^{-3}$, $5.271 \times 10^{-3}$, $1.123 \times 10^{-2}$ and $4.990 \times 10^{-3}$ on the CPU at that commit; the devices round differently, and the runs drift apart only in the third digit.
The preset now trains on the CPU unless `accelerator` says otherwise.

## Not compared

- Tables 2 and 3 (T9 in 3D, T10 in 4D) are ICNN results, and the penalty MLP is 2D only (`PenaltyMLP` does not support T9 or T10).
  `Box` samples its boundary in any dimension, so the shared loop can run them; they belong to the ICNN of #13, or to 0.2.
- Tables 4 to 6 (average error against the number of collocation points) and Figures 2 to 6 (T1, T3, T2, T5) are ICNN results.

## Rerun

```sh
just push
just remote-bg pmlp-t1 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_penalty_mlp.py T1
# when it has finished:
just remote-bg pmlp-t6 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_penalty_mlp.py T6
```

The two diagnostics of T1 (Findings 2 and 4) write their CSVs to the data directory, not to the receipts:

```sh
just remote-bg pmlp-fit-tol-t1 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_penalty_mlp.py T1 --fit-tol 1e-7 --out /jczhang02/maesolvers-data/data/penalty_mlp_t1_fit_tol.csv
just remote-bg pmlp-bfgs-t1 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python benchmarks/run_penalty_mlp.py T1 --optimizer bfgs --out /jczhang02/maesolvers-data/data/penalty_mlp_t1_bfgs.csv
```

Each takes $8$ to $10$ minutes on a100server with ten CPU threads; run them one after the other, to stay within $16$ threads.
