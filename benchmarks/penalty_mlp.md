# Penalty MLP: published versus ours

`PenaltyMLP` with the paper's defaults, seeds 0 to 9, against Nystrom and Vestberg (2023), arXiv 2110.03310v3, read from the PDF.
**Level (ADR 0002): figure only**, on T1 against Figure 1: the paper prints no table for this method (Finding 1).
On T6 we ran the method next to Table 1, which belongs to the paper's other method; that comparison is context, not a reproduction.

Receipts: `penalty_mlp_t1.csv` and `penalty_mlp_t6.csv`, written by `run_penalty_mlp.py` on a100server at commit b4596fc, CPU, one thread per seed; every seed is an MLflow run in `maesolvers-penalty-mlp-T1` and `maesolvers-penalty-mlp-T6`, with its training curve.
Bands are the median and the 5% and 95% quantiles over the ten seeds (ADR 0007); no seed failed, and every seed stopped at the 5000-step budget (`stop = max_steps`).

## Finding 1: the paper's tables belong to the ICNN, not to the penalty MLP

The paper has two methods: the penalty MLP of Section 3.1 and an input convex network (ICNN) of Section 3.2.
The penalty MLP is used once, in Example 4.1 on T1, and its only result is Figure 1.
Every later example uses the ICNN:

- Section 4, before Example 4.1: the first example uses the penalty method, "in the subsequent ... examples" the input convex networks.
- Example 4.6 (T6, Table 1): "We use the same network architecture as in Example 4.2", and Example 4.2 is the ICNN (5 x 10 softplus, 563 parameters, loss `Ee + 100 Eb` with the boundary points also in `Ee`, BFGS).
  Figure 7, the same example without noise after 5000 BFGS iterations, has a colorbar that ends at about 8.5e-6, the `stdev = 0` row of Table 1.
- Section 4.7 (Tables 2 and 3, T9 and T10): "we use input convex networks to solve higher-dimensional equivalents"; Section 3.1 says the penalty "essentially only works in the two-dimensional case".

The research survey's plan (reproduce the penalty MLP against Table 1 and the 3D tables) is therefore wrong, and issue #12 inherited it.
The finding is on #12 and #13: the ICNN of #13 can reach the statistical level against Table 1 on T6, and against Tables 2 and 3 once it runs in 3D and 4D.
`papers.md` is corrected.

## T1 (Example 4.1): figure only

Figure 1 plots the absolute error of the trained network on the unit square.
Read from the rendered page (p. 10): the colorbar runs from 0 to about 5.7e-3 (its top tick is 0.0054), most of the square is below 1.5e-3, and the largest errors are at the boundary, near the corners `(0, 0)`, `(1, 0.9)` and `(1, 1)`.
The paper gives no iteration count for this run, and restarted BFGS with Adam when the line search failed.

| Measure | Ours, median [5%, 95%] |
|---|---|
| max, shared 201 x 201 grid (`evaluate`) | 7.59e-3 [4.69e-3, 2.22e-2] |
| RMS, shared grid (`evaluate`) | 1.14e-3 [6.11e-4, 4.95e-3] |
| max, 10^6 random interior points (the paper's measure) | 7.40e-3 [4.67e-3, 2.22e-2] |
| mean, 10^6 random interior points (the paper's measure) | 8.10e-4 [4.42e-4, 4.11e-3] |
| max on the boundary, 10^5 random points | 7.59e-3 [4.69e-3, 2.22e-2] |

Verdict: the figure's largest error, about 5.7e-3, lies inside our band of max errors, and our errors, like the figure's, are largest at the boundary.
That is agreement at the figure-only level; the figure gives one run and a colour scale, so no closer statement is possible.
Two seeds (1 and 8) are clearly worse than the rest: their loss is still above 1e3 after 5000 steps and their max errors are 1.8e-2 and 2.6e-2.

## T6 (Example 4.6): our runs next to the ICNN's Table 1

| Measure | Table 1, `stdev = 0` (ICNN, one run) | PenaltyMLP, median [5%, 95%] |
|---|---|---|
| max error, 10^6 random interior points | 8.58e-6 | 5.42e-3 [3.29e-3, 1.00e-2] |
| average error, 10^6 random interior points | 1.17e-6 | 6.49e-4 [3.59e-4, 1.53e-3] |
| max, shared grid (`evaluate`) | | 5.31e-3 [3.18e-3, 9.25e-3] |
| RMS, shared grid (`evaluate`) | | 8.91e-4 [5.44e-4, 2.00e-3] |
| max on the boundary, 10^5 random points | | 5.43e-3 [3.29e-3, 1.01e-2] |

The penalty MLP is about 600 times less accurate than the paper's ICNN on T6, in line with the paper's own comparison on T1 (errors of order 1e-3 for the penalty MLP in Figure 1, 1e-5 for the ICNN in Figure 2).
This is a comparison of two methods, with no reproduction verdict.

## Finding 2: the error is the boundary fit

In every seed of both problems the max error at the boundary points is at least the max error at the 10^6 interior points, and at most 6% above it (the CSVs per seed): the error is largest on the boundary, and the interior points closest to it come near.
On the boundary the ansatz `G + D N` is `G` up to the small misfit of `D`, so the max error is the misfit of `G`, the one-layer network of 20 tanh units fitted to `g`, and training `N` cannot reduce it.
The fit stops where the paper's does, when the largest gradient component of the mean squared error is at most 1e-5 (SciPy's default); for the mean squared error of a fit to values of order 1, this is reached early.
In a diagnostic outside the receipts, a tolerance of 1e-7 brought the max boundary misfit of `G` on T1 from 2.9e-3 to 1.3e-2 down to 1.3e-3 to 3.5e-3 over three seeds, at 20000 iterations; the preset keeps the paper's setting.
The interior error is smaller: the mean error is about a tenth of the max (tables above).

## Finding 3: the penalty does not keep the network convex everywhere

The penalty acts at the collocation points only.
On T1, `evaluate` finds a negative Hessian eigenvalue at some points of the shared grid in 7 of 10 seeds, at up to 5.4% of the points (seeds 1 and 8, the two that trained worst), and at most 0.43% in the others; on T6 only seed 3 has any (0.21%).
The FD solvers report `nonconvex_fraction` 0 on T1.

## Conventions that differ

- **Error measure.** The paper measures at 10^6 random interior points (Section 4.6); the library's primary error is on the shared evaluation grid (ADR 0007), and the receipts report both.
  Our 10^6 points are uniform in the domain, drawn by rejection from one fixed generator seed (2023) for every run.
  The two max errors differ by at most 6% on T1, whose shared grid includes the boundary of the box, and the grid's is up to 13% lower on T6, whose shared grid keeps interior points only: the max error lies on the boundary, which random interior points approach more closely than the grid's interior points.
- **Optimizer.** L-BFGS with 100 curvature pairs and a strong Wolfe line search, one iteration per step, for SciPy's full BFGS; no Adam restarts on a failed line search.
- **Budget.** 5000 L-BFGS steps; the paper gives no count for Example 4.1 (5000 BFGS iterations is the budget of its Example 4.6).
  No seed met the gradient tolerance within it; the loss was still decreasing.
- **Initial weights.** Standard normal, as the paper says; it does not give the standard deviation, and 1 is TensorFlow's default.
- **Distance network `D`.** Fitted to the exact distance on both problems; the paper does so on the unit square and points to Berg and Nystrom (2018) for other domains.
- **Float64** throughout, as ADR 0008 requires; the paper does not say which precision it used.

## Device

The receipts train on the CPU.
The networks are small, so a training step is bound by the launch of many small kernels: on a100server one A100 took about 70 ms per step for one process and about 200 ms per step for each of four processes sharing it, while one CPU thread took about 70 ms per step and ten seeds ran side by side in about 8 minutes.
Seeds 0 to 4 of T6 also ran on an A100 (MLflow, commit 6bb1f9e): their shared-grid max errors are 4.810e-3, 6.792e-3, 5.256e-3, 1.124e-2 and 4.991e-3, against 4.809e-3, 6.795e-3, 5.271e-3, 1.123e-2 and 4.990e-3 on the CPU; the devices round differently, and the runs drift apart only in the third digit.

## Not compared

- Tables 2 and 3 (T9 in 3D, T10 in 4D) are ICNN results, and the penalty MLP is 2D only (`PenaltyMLP` does not support T9 or T10).
  `Box` samples its boundary in any dimension, so the shared loop can run them; they belong to the ICNN of #13, or to 0.2.
- Tables 4 to 6 (average error against the number of collocation points) and Figures 2 to 6 (T1, T3, T2, T5) are ICNN results.

## Rerun

```sh
just push
just remote-bg pmlp-t6 env CUDA_VISIBLE_DEVICES= python benchmarks/run_penalty_mlp.py T6
just remote-bg pmlp-t1 env CUDA_VISIBLE_DEVICES= python benchmarks/run_penalty_mlp.py T1
```

Each takes about 8 minutes on a100server with ten CPU threads.
