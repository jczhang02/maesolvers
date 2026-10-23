# QS-PINN against its baselines

QS-PINN (`docs/methods/qs_pinn.md`) has no source paper and no published table: JC Zhang's note (`writing/notes/qs-pinn/main.tex` in jczhang02/monge-ampere-equations) states that none of its claims has training support yet.
So this file holds no reproduction and no ADR 0002 level; it compares QS-PINN with the baselines the note names (Section 7), on the note's problems (Table 2), over seeds 0 to 9, and runs QS-PINN alone on the catalogue's T10 to see what limits it in 4D (Finding 5).

## Receipt

- `qs_pinn.csv`, one row per method, problem and seed (160 rows), written on a100server by `run_qs_pinn.py` with 8 worker processes of one CPU thread each, on the CPU, in two runs.
  The baselines' 110 rows (`mlp`, `penalty_mlp`, `icnn`, `icnn_quadratic`) are from the first run, at commit 0526410, in 6924 s of wall time: `just remote-bg qs-pinn python benchmarks/run_qs_pinn.py --workers 8 --out /jczhang02/maesolvers-data/data/qs_pinn.csv`.
  The `qs` rows and the T10 rows (`qs`, `qs_ratio_0.9`) are from the second run, at commit 249e131 (review round 1 of #69, which keeps `alpha` positive), clean checkout, in 1300 s, merged into a copy of the first receipt: `just remote-bg qs-pinn-r1 env CUDA_VISIBLE_DEVICES= python benchmarks/run_qs_pinn.py --methods qs qs_ratio_0.9 --workers 8 --name r1 --out /jczhang02/maesolvers-data/data/qs_pinn-r1.csv`; the merge keeps the other rows, which agree with the first receipt to every byte.
  The script's docstring lists the columns; `python benchmarks/run_qs_pinn.py --summary` prints every cell of the tables below from the receipt (the medians, the smallest `eigmin`, the seeds of each count, the stops with their seeds and steps).
- Every seed is one MLflow run, with the parameter `method` and its training curve, in `maesolvers-qs_pinn-T8` (P1 and T8, tagged `g=0.5` and `g=0`), `maesolvers-qs_pinn-T12` or `maesolvers-qs_pinn-T10`.
  The first run's are named `<method> seed=<s>`; they started from the clean checkout of 0526410, and the server checkout moved to c35c01e while they ran, so the later ones carry that tag (the two commits differ only in `benchmarks/comparison.py`'s `unsupported_reason` and its test, which the runner does not call).
  The second run's are named `r1 <method> seed=<s>`, all at 249e131.
  The `qs` runs of the first version, with a bare trained `alpha`, stay in MLflow under their first names; their numbers are quoted below where they differ.
- The QS presets' rows of the comparison table (`comparison.csv`, from `run_comparison.py` at 249e131, also one CPU thread per seed) agree with this receipt to every digit: `DirichletQSPINN` on T8 and T10 and `TransportQSPINN` on T12, every seed's max error; so do the baselines `DirichletICNN` and `PenaltyMLP` on T8 and `TransportICNN` on T12.

## Problems

- **P1**, the note's P1: the unit disk, `f = 1`, `g = 1/2`, convex solution `|x|^2 / 2`, concave solution `1 - |x|^2 / 2` (the note's Example 1.1).
  It is the catalogue's T8 plus 1/2 and has no T id of its own, so it is defined in the runner (`P1`), not in the catalogue.
- **T8**, the catalogue's problem: the same with `g = 0`, convex solution `(|x|^2 - 1) / 2`, concave `(1 - |x|^2) / 2`.
  The constant changes no error (the evaluation grid, the threshold, the collocation points are the same), but it changes where the QS form starts (Finding 1), so both are run.
- **T12**, the note's P3: the transport of the unit disk onto the ellipse centred at `(3.5, 0)` with semi-axes 2 and 1/2, solution `x1^2 + 3.5 x1 + x2^2 / 4` up to a constant.
- **T10**, the catalogue's `exp(|x|^2 / 2)` on `(0, 1)^4`, with `x0` the centre: QS-PINN alone, with its default `ratio = 0.6` (`qs`) and with `ratio = 0.9` (`qs_ratio_0.9`), a diagnostic, not a tuning (Finding 5).
- The note's P2, `sqrt(1 + |x|^2)` on `(-1, 1)^2`, has no T id and is not trained here; `tests/test_qs_pinn.py` checks that it has the exact QS form of the note's Example 4.5.

## Methods

The same seed gives the same collocation points to every method with the same sampler (the fingerprints are equal), and on each problem the methods share every part but the network, except the penalty MLP:

| Method | P1 and T8 | T12 |
|---|---|---|
| `qs` | `DirichletQSPINN()` | `TransportQSPINN()` |
| `mlp` | the parts of `DirichletICNN()` with `MLP(width=10, depth=5, activation="tanh")`, 481 parameters | the parts of `TransportICNN()` with `IdentityStart(network=MLP(width=10, depth=4, activation="tanh"))` |
| `penalty_mlp` | `PenaltyMLP()`, its own parts (ansatz, convexity penalty, L-BFGS) | not run: its ansatz is Dirichlet |
| `icnn` | `DirichletICNN()` | `TransportICNN()` |
| `icnn_quadratic` | the parts of `DirichletICNN()` with `ICNN(alpha=0.25)` | the parts of `TransportICNN()` with `IdentityStart(network=ICNN(depth=4, alpha=0.25))` |

So on P1 and T8 every method but the penalty MLP minimises the summed residual plus 100 times the mean squared boundary misfit with dense BFGS for up to 10000 steps, and on T12 every method minimises the mean residual plus the Hausdorff-type transport term with 200 Adam steps and then L-BFGS, 2000 steps in all (`docs/methods/icnn.md`).
The plain MLP takes the activation of the QS form's network, tanh, so that the two differ in the QS structure alone.
On T10, `qs` is `DirichletQSPINN()` and `qs_ratio_0.9` is `DirichletQSPINN(ratio=0.9)`, with the same points and start.

## P1 (T8 with `g = 1/2`)

Threshold: 1% of the oscillation of the exact solution on the shared grid, 5.00e-3, as in the comparison table.

| Method | Max error, median [5%, 95%], seeds below the threshold | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Seeds on the concave solution | Time (s), median | Stops |
|---|---|---|---|---|---|---|---|
| `qs` | 2.22e-16 [2.22e-16, 2.22e-16], 10/10 | 0 | 1.00 | 10 | 0 | 0.653 | converged 10 (at step 0) |
| `mlp` | 6.31e-1 [3.46e-5, 2.47e0], 2/10 | 8 | -1.44e3 | 3 | 1 (seed 7) | 71.2 | converged 4, stalled 6 |
| `penalty_mlp` | 5.26e-3 [4.10e-3, 9.37e-3], 3/10 | 1 (seed 1) | -1.30e-1 | 9 | 0 | 453 | max_steps 10 |
| `icnn` | 7.72e-6 [3.74e-6, 1.34e-5], 10/10 | 0 | 1.00 | 10 | 0 | 934 | converged 4, max_steps 4, stalled 2 |
| `icnn_quadratic` | 4.96e-6 [3.50e-6, 1.11e-5], 10/10 | 0 | 1.00 | 10 | 0 | 807 | converged 7, max_steps 3 |

## T8 (`g = 0`)

| Method | Max error, median [5%, 95%], seeds below the threshold | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Seeds on the concave solution | Time (s), median | Stops |
|---|---|---|---|---|---|---|---|
| `qs` | 1.72e-7 [4.03e-8, 3.82e-7], 10/10 | 0 | 1.00 | 10 | 0 | 48.5 | converged 10 |
| `mlp` | 5.28e-1 [1.97e-1, 1.94e0], 1/10 | 9 | -1.55e3 | 2 | 1 (seed 7) | 48.4 | converged 3, stalled 7 |
| `penalty_mlp` | 5.04e-3 [3.90e-3, 8.96e-3], 5/10 | 1 (seed 1) | -5.83e-2 | 9 | 0 | 418 | max_steps 10 |
| `icnn` | 6.61e-6 [2.88e-6, 1.22e-5], 10/10 | 0 | 1.00 | 10 | 0 | 870 | converged 8, max_steps 2 |
| `icnn_quadratic` | 7.31e-6 [4.09e-6, 1.18e-5], 10/10 | 0 | 1.00 | 10 | 0 | 866 | converged 5, max_steps 4, stalled 1 |

## T12

Threshold: 6.93e-2; the max and RMS errors of `u` are taken after both sides' means are subtracted (ADR 0007), and the error of the map `grad u` is the root mean square of `|grad u - grad u_exact|` on the shared grid (`gradient_map_error`, as `benchmarks/icnn.md` reports it), with `grad u` and `grad u_exact` both taken by central differences of `u` on that grid (`gradient_fd`).
`into` and `onto` are the two halves of the final transport term of the loss (`HausdorffLoss`), at the training's boundary and target points.

| Method | Max error, median [5%, 95%], seeds below the threshold | RMS error, median | Map error (RMS), median | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Final loss, median | `into`, `onto`, medians | Time (s), median |
|---|---|---|---|---|---|---|---|---|---|
| `qs` | 1.32e-3 [8.80e-4, 3.81e-3], 10/10 | 4.94e-4 | 1.20e-3 | 0 | 4.95e-1 | 10 | 1.36e-4 | 5.65e-5, 8.03e-5 | 244 |
| `mlp` | 1.82e-3 [9.07e-4, 3.49e-3], 10/10 | 6.95e-4 | 1.90e-3 | 0 | 4.95e-1 | 10 | 1.35e-4 | 5.68e-5, 7.94e-5 | 182 |
| `icnn` | 1.72e-3 [1.33e-3, 4.14e-3], 10/10 | 7.85e-4 | 1.80e-3 | 0 | 4.97e-1 | 10 | 1.37e-4 | 5.70e-5, 8.02e-5 | 221 |
| `icnn_quadratic` | 1.82e-3 [9.09e-4, 3.78e-3], 10/10 | 7.12e-4 | 1.69e-3 | 0 | 4.97e-1 | 10 | 1.37e-4 | 5.68e-5, 7.98e-5 | 231 |

The published T12 numbers (Caboussat-Peruso 2025, Table 2: the L2 error of the map, 2.67e-3 for L-BFGS, as a mean over ten runs) are compared with `TransportICNN` in `benchmarks/icnn.md` (T12), with the caveat that the paper's L2 error may be the L2 norm over the disk rather than this root mean square; read the map errors here against that comparison, not against 2.67e-3 directly.

Every T12 run stopped at its step budget; no seed of any method failed on any problem (no `"diverged"` stop, no non-finite error).
`summary()` prints the band of the RMS error, the map error, the loss and `into` and `onto` too; the medians above are its first numbers.

## T10 (4D), QS-PINN alone

Threshold: 6.39e-2, 1% of `e^2 - 1`.

| Method | Max error, median [5%, 95%], seeds below the threshold | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Final loss, median | Time (s), median | Stops |
|---|---|---|---|---|---|---|---|
| `qs` (`ratio = 0.6`) | 1.11e0 [7.56e-1, 1.42e0], 0/10 | 5 (seeds 1, 2, 5, 7, 8) | -6.30e6 | 0 | 1.33e2 | 215 | stalled 10, after 645 to 1359 steps |
| `qs_ratio_0.9` | 9.26e-1 [3.98e-1, 2.28e0], 0/10 | 2 (seeds 6, 7) | -8.18e4 | 0 | 2.67e1 | 216 | stalled 10, after 654 to 1877 steps |
"Seeds on the concave solution" counts the seeds whose max distance to the concave solution is below the threshold.
"Seeds with `max |det D^2 u - f| < min f`" counts the seeds that pass the residual half of the note's Corollary 5.1 on the shared grid: a statement about grid points, not a proof, since the supremum between them is not bounded.

## Finding 1: on P1 the QS form starts at the exact solution

`alpha` starts at `f(x0)^(1/(2d)) / sqrt(2)` and `b`, `c` and the output layer of the matrix network at 0 (the note's Section 6), so the QS form starts from `u = f(x0)^(1/d) |x - x0|^2 / 2`.
On P1, `f = 1`, `x0 = 0` and the solution is `|x|^2 / 2`: the start is the solution, the first loss is 6.7e-28 and every seed stops `converged` before its first step, with a max error of 2.2e-16.
The same holds for every problem whose solution is `f(x0)^(1/d) |x - x0|^2 / 2`, so P1 says nothing about how QS-PINN trains; `tests/test_qs_pinn.py` (`test_the_form_starts_from_alpha_squared_times_the_distance_squared`) pins the start.
T8 differs from P1 by the constant `-1/2` alone, which `c` must learn: there QS-PINN reaches 1.72e-7 [4.03e-8, 3.82e-7] in 299 to 730 BFGS steps (48.5 s in the median), against the ICNN's 6.61e-6 [2.88e-6, 1.22e-5] in about 870 s, about 38 times more accurate and 18 times faster, and every seed stops `converged`, is convex on the grid and passes the residual half of the certificate.
The first version, with a bare trained `alpha`, reached 4.65e-8 [8.45e-9, 3.93e-7] in 223 to 1099 steps, two seeds stopping `stalled`: `alpha = alpha_start exp(s)` changes the path BFGS takes, since the gradient in `s` is `alpha` times the gradient in `alpha`, not the problem.
T12 (Finding 3) is the problem on which the QS form starts away from the solution.

## Finding 2: the plain MLP does not select the convex branch, and its spurious stationary point is `u = g`

Under the same loss, points and optimizer as the ICNN, the plain tanh MLP succeeds in 2 seeds of 10 on P1 and 1 on T8:

- **Concave branch.** Seed 7 reaches the concave solution on both problems (max distance to it 1.05e-4 on P1 and 1.11e-4 on T8, Hessian negative definite at every grid point): the note's Example 1.1, now in the shared protocol. It also passes the residual half of the certificate (`det D^2 u > 0` at every grid point), which shows why Corollary 5.1 needs the Hessian to be positive definite at one point, the half the QS form gives by construction.
- **`u` constant.** Seeds 0, 2 and 8 on P1, and 0, 2 and 8 on T8, stop `converged` within 8 to 30 steps at a function with max error 0.500 and loss 3.4e3: `u` equals `g` to 4e-5 on the boundary and is nearly flat inside, so `det D^2 u` is near 0 and the residual near `-1` everywhere. The gradient of `det` in the Hessian is its cofactor matrix, which vanishes at a zero Hessian, so `u = g` is a stationary point of the loss: a spurious critical point of the kind the note cites (Section 2). The QS form has the same kind of point at `alpha = 0`, where `A = 0` and `u = b . x + c`: `u` depends on `alpha` through `alpha^2` only, so every gradient through `A` (in `alpha` and in the matrix network) vanishes there, and with the plane `b . x + c` fitted to `g` it is a stationary point of the loss (on T8, `g = 0`, at `b = c = 0`); `alpha = alpha_start exp(s)` keeps `alpha` positive, so the form cannot reach it, only approach it as `s` goes to minus infinity, and its Hessian at `x0`, `2 A^T A` with `A = alpha (I + B)` invertible, stays positive definite. No QS run here approached it: every `qs` seed on P1, T8 and T12 ends with `eigmin` above 0.49 on the grid.
- **The rest** stall at errors of 0.44 to 2.7, non-convex at 16% to 95% of the grid, with smallest Hessian eigenvalues down to `-1.6e3`.

Seeds 3 and 4 on P1 and 6 on T8 reach the convex solution (6.9e-6 to 9.0e-5).
"Seeds on the concave solution" in the tables counts the seeds within the threshold of the concave solution, seed 7 alone; seeds 5, 6 and 9 on P1, and 5 and 9 on T8, lie nearer the concave solution than the convex one but far from both (max distances 0.43 to 2.25).
The note's demonstration (Section 1: a 2-32-32-1 tanh MLP, Adam, a mean loss with boundary weight 10, points resampled every step) put all ten seeds on the concave solution; under this protocol one seed of ten goes there and most stop elsewhere, so the protocol, not only the network, decides which wrong answer the plain MLP gives.

## Finding 3: on T12 the four networks reach the same error, the level the sampling leaves

QS-PINN starts on T12 from `u = |x|^2 / 2`, whose map is the identity, and needs no fit to it; the ICNN and the plain MLP start from a fitted identity map, so all four methods start from about the same map.
After 2000 steps their medians lie within a factor 1.4 of each other in the max error (1.32e-3 to 1.82e-3) and 1.6 in the map error (1.20e-3 to 1.90e-3), with overlapping bands, and their final losses within 2% (1.35e-4 to 1.37e-4), as are the two halves `into` and `onto` of the transport term.
That loss is the level the exact map itself leaves at independently drawn boundary and target points (1.1e-4 to 1.7e-4 for seeds 0 to 9, `benchmarks/icnn.md`, Finding 3): under this sampling the transport term, not the network, sets the error, and T12 does not separate the methods.
QS-PINN has the smallest medians, but its band overlaps the others', and the first version's (max error 1.52e-3 [1.01e-3, 3.92e-3]) shows how much the medians move with the parametrisation alone.
The plain MLP, not convex by construction, is convex on the grid in every seed: the problem's solution is a quadratic with Hessian `diag(2, 1/2)`, far from any non-convex function, and the identity start is convex too.

## Finding 4: the QS form is not convex by construction away from `x0`

On T12 every QS seed is convex on the grid (`eigmin` 0.495 to 0.4997, the exact solution's 0.5) and passes the residual certificate there.
The first version, with a bare `alpha`, did not: its seed 1 had a negative smallest Hessian eigenvalue, `-3.93e-2`, at 1 point of the 31397 of the shared grid, and a largest residual of 1.05 against `min f = 1`, though its errors were in the band of the others.
The form guarantees the structure at `x0` (Proposition 3.3), not convexity elsewhere: only the note's Theorem 4.3 or Corollary 5.1 does, and the training checks neither.
The comparison table shows it away from the note's problems: on T2, whose gradient blows up at a corner, `DirichletQSPINN` is non-convex at 18.7% of the grid in the median, with errors of 0.34 to 5.1 and one seed (9) diverging; on T10 (Finding 5) five seeds of ten are non-convex at some grid points, with Hessian eigenvalues down to `-6.3e6`.

## Finding 5: on T10 the training stalls before the bound on `B` matters

The bound `ratio` on `|B|_F` limits which `A` the form can take at a point.
For `exp(|x|^2 / 2)` on `(0, 1)^d` with `x0` the centre, the canonical matrix of the note's Lemma 3.2, `A = M^(1/2)`, fits inside `|A / alpha - I|_F < ratio` for one `alpha` only if `ratio` exceeds 0.28 in 2D (T1), 0.46 in 3D (T9) and 0.64 in 4D (T10) (`python benchmarks/run_qs_pinn.py --canonical`; `M` by 20-point Gauss-Legendre quadrature on grids of 21, 11 and 7 points per side, corners included).
The spread of the singular values of that `A` at a point is at most 1.41, 1.59 and 1.76, well inside the factor 4 that `ratio = 0.6` allows: the Frobenius ball, not the spread, is the constraint that grows with `d`, as `|B|_F` can reach `sqrt(d)` times `|B|_op`.
So `ratio = 0.6`, chosen on T12 (Conventions), holds the canonical representation of T1 and T9 but not of T10; `A` is not unique (only `|A (x - x0)|` enters `u`), so this does not prove that no `A` within 0.6 represents T10.
More room does not cure T10: with `ratio = 0.9`, which holds the canonical representation, every seed still stops `stalled` (SciPy's line search failing) after 654 to 1877 steps, at a max error of 9.26e-1 [3.98e-1, 2.28e0] against 1.11e0 [7.56e-1, 1.42e0] at 0.6, all 20 far above the threshold of 6.39e-2.
On T10 the optimizer's stall, not the bound, sets the error.
A review of the first version found the same with a bare `alpha` (MLflow `maesolvers-qs_pinn-T10`, runs `review ratio=0.9 seed=<s>` and `review ratio=0.6 seed=<s>`, seeds 0 to 2: 0.61, 0.49 and 1.54 at 0.9, 1.09, 1.19 and 0.78 at 0.6, every seed `stalled`).
`DirichletICNN` does worse on T10 (README, Comparison: 9.09e2 in the median), so this is a limit of the training in 4D, not of the QS form alone; `ratio` is not tuned here.

## Conventions

- **No level.** There is no published number to place in a band, so ADR 0002's levels do not apply; the threshold for "seeds below" is the comparison table's (1% of the exact oscillation on the shared grid), not a verdict.
- **The loss.** The note's loss (Section 6) is a mean of the squared residual and of the squared boundary misfit; the Dirichlet runs use the ICNN preset's loss, the residual summed over the 3400 interior and boundary points plus 100 times the mean squared misfit, so that the comparison isolates the network (`docs/methods/qs_pinn.md`). Whether the ICNN's sum or a mean is the right reading of its paper is open (#13); the QS presets follow the current code.
- **`alpha` of the ICNN plus a quadratic.** 0.25, half the smallest exact Hessian eigenvalue of T8 (1) and T12 (1/2), so the exact solutions stay representable; we chose it knowing them, an advantage a user solving an unknown problem does not have.
- **`ratio` of the QS form.** 0.6 was also chosen knowing an exact solution: T12's canonical matrix needs `|B|_F` of at least 0.447 (`docs/methods/qs_pinn.md`, Parameters), and 0.6 leaves a margin above it; the note gives no value. On the exponential problems the same arithmetic gives 0.28 in 2D (T1), 0.46 in 3D (T9) and 0.64 in 4D (T10), so the default is justified in 2D and 3D only (Finding 5).
- **The QS form is a network part.** ADR 0004 lists QS among the losses (`a loss (penalty, QS)`); here it is a network part, `QuadraticForm`, since it is the representation of `u`, not an objective, and it trains under the ICNN presets' losses.
- **The collocation points** are drawn from the seed by each sampler and fingerprinted, not stored (#52); the methods that share a sampler share the points.
- **Device.** One CPU thread per seed, every method, as in the comparison table; `icnn_t12.csv` trained `TransportICNN` on a GPU and differs from the `icnn` rows of T12 here by rounding (`benchmarks/comparison.md`).
- **Not measured:** the note's derivative-bound certificate (Theorem 4.3, which needs bounds on the derivatives of `A`), its Lipschitz regulariser, its ablations, and the error bound of its Proposition 2.1. The final `alpha` is in each run's `Result.info["network"]`, not in the receipt.
- **The note's T12 metrics.** Its Section 7 asks for the boundary-map error, the L2 error and a comparison with the published results on P3: the T12 table gives `into` and `onto` (the boundary-map terms of the loss), the RMS error of `u` and of `grad u`, and points to `benchmarks/icnn.md` for the published Table 2.

## Rerun

```sh
just push
just remote-bg qs-pinn python benchmarks/run_qs_pinn.py --workers 8 \
    --out /jczhang02/maesolvers-data/data/qs_pinn.csv
# to rerun some methods only, copy the receipt to that --out path on
# a100server first and add --methods (as for the second run above)
# when the log ends with "wrote ...":
scp a100server:/jczhang02/maesolvers-data/data/qs_pinn.csv benchmarks/
python benchmarks/run_qs_pinn.py --summary
python benchmarks/run_qs_pinn.py --canonical   # Finding 5's bounds; seconds, nothing trained
```

Do not `just push` while it runs: the server checkout moves, and the later runs carry the new commit.
