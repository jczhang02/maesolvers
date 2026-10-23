# QS-PINN against its baselines

QS-PINN (`docs/methods/qs_pinn.md`) has no source paper and no published table: JC Zhang's note (`writing/notes/qs-pinn/main.tex` in jczhang02/monge-ampere-equations) states that none of its claims has training support yet.
So this file holds no reproduction and no ADR 0002 level; it compares QS-PINN with the baselines the note names (Section 7), on the note's problems (Table 2), over seeds 0 to 9.

## Receipt

- `qs_pinn.csv`, one row per method, problem and seed (140 rows), written on a100server by `run_qs_pinn.py` in 6924 s of wall time with 8 worker processes of one CPU thread each, on the CPU: `just remote-bg qs-pinn python benchmarks/run_qs_pinn.py --workers 8 --out /jczhang02/maesolvers-data/data/qs_pinn.csv`, then copied here.
  The script's docstring lists the columns; `python benchmarks/run_qs_pinn.py --summary` prints every cell below from the receipt.
- Every seed is one MLflow run, named `<method> seed=<s>` with the parameter `method`, in `maesolvers-qs_pinn-T8` (P1 and T8, tagged `g=0.5` and `g=0`) or `maesolvers-qs_pinn-T12`, with its training curve.
  The runs started from the clean checkout of commit 0526410; the server checkout moved to c35c01e while they ran, so the later runs carry that tag.
  The two commits differ only in `benchmarks/comparison.py`'s `unsupported_reason` and its test, which the runner does not call.
- The QS presets' rows of the comparison table (`comparison.csv`, from `run_comparison.py` at 0526410, also one CPU thread per seed) agree with this receipt to every digit: `DirichletQSPINN` on T8 and `TransportQSPINN` on T12, and the baselines `DirichletICNN` and `PenaltyMLP` on T8 and `TransportICNN` on T12, every seed's max error.
- A diagnostic before the receipt, `qs` on T8 and T12 with seeds 0 and 1 (MLflow runs `diagnostic qs seed=<s>`, `/jczhang02/maesolvers-data/data/qs_diag.csv`), gave the receipt's numbers for those seeds; nothing was changed after it.

## Problems

- **P1**, the note's P1: the unit disk, `f = 1`, `g = 1/2`, convex solution `|x|^2 / 2`, concave solution `1 - |x|^2 / 2` (the note's Example 1.1).
  It is the catalogue's T8 plus 1/2 and has no T id of its own, so it is defined in the runner (`P1`), not in the catalogue.
- **T8**, the catalogue's problem: the same with `g = 0`, convex solution `(|x|^2 - 1) / 2`, concave `(1 - |x|^2) / 2`.
  The constant changes no error (the evaluation grid, the threshold, the collocation points are the same), but it changes where the QS form starts (Finding 1), so both are run.
- **T12**, the note's P3: the transport of the unit disk onto the ellipse centred at `(3.5, 0)` with semi-axes 2 and 1/2, solution `x1^2 + 3.5 x1 + x2^2 / 4` up to a constant.
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

## P1 (T8 with `g = 1/2`)

Threshold: 1% of the oscillation of the exact solution on the shared grid, 5.00e-3, as in the comparison table.

| Method | Max error, median [5%, 95%], seeds below the threshold | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Seeds on the concave solution | Time (s), median | Stops |
|---|---|---|---|---|---|---|---|
| `qs` | 2.22e-16 [2.22e-16, 2.22e-16], 10/10 | 0 | 1.00 | 10 | 0 | 0.593 | converged 10 (at step 0) |
| `mlp` | 6.31e-1 [3.46e-5, 2.47e0], 2/10 | 8 | -1.44e3 | 3 | 1 (seed 7) | 71.2 | converged 4, stalled 6 |
| `penalty_mlp` | 5.26e-3 [4.10e-3, 9.37e-3], 3/10 | 1 (seed 1) | -1.30e-1 | 9 | 0 | 453 | max_steps 10 |
| `icnn` | 7.72e-6 [3.74e-6, 1.34e-5], 10/10 | 0 | 1.00 | 10 | 0 | 934 | converged 4, max_steps 4, stalled 2 |
| `icnn_quadratic` | 4.96e-6 [3.50e-6, 1.11e-5], 10/10 | 0 | 1.00 | 10 | 0 | 807 | converged 7, max_steps 3 |

## T8 (`g = 0`)

| Method | Max error, median [5%, 95%], seeds below the threshold | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Seeds on the concave solution | Time (s), median | Stops |
|---|---|---|---|---|---|---|---|
| `qs` | 4.65e-8 [8.45e-9, 3.93e-7], 10/10 | 0 | 1.00 | 10 | 0 | 52.9 | converged 8, stalled 2 (seeds 1, 2) |
| `mlp` | 5.28e-1 [1.97e-1, 1.94e0], 1/10 | 9 | -1.55e3 | 2 | 1 (seed 7) | 48.4 | converged 3, stalled 7 |
| `penalty_mlp` | 5.04e-3 [3.90e-3, 8.96e-3], 5/10 | 1 (seed 1) | -5.83e-2 | 9 | 0 | 418 | max_steps 10 |
| `icnn` | 6.61e-6 [2.88e-6, 1.22e-5], 10/10 | 0 | 1.00 | 10 | 0 | 870 | converged 8, max_steps 2 |
| `icnn_quadratic` | 7.31e-6 [4.09e-6, 1.18e-5], 10/10 | 0 | 1.00 | 10 | 0 | 866 | converged 5, max_steps 4, stalled 1 |

## T12

Threshold: 6.93e-2; the max error of `u` is taken after both sides' means are subtracted (ADR 0007), and the error of the map `grad u` is the root mean square of `|grad u - grad u_exact|` on the shared grid (`gradient_map_error`, as `benchmarks/icnn.md` reports it).

| Method | Max error, median [5%, 95%], seeds below the threshold | Map error (RMS), median | Seeds non-convex on the grid | Smallest `eigmin` | Seeds with `max |det D^2 u - f| < min f` | Final loss, median | Time (s), median |
|---|---|---|---|---|---|---|---|
| `qs` | 1.52e-3 [1.01e-3, 3.92e-3], 10/10 | 1.50e-3 | 1 (seed 1) | -3.93e-2 | 9 | 1.33e-4 | 223 |
| `mlp` | 1.82e-3 [9.07e-4, 3.49e-3], 10/10 | 1.90e-3 | 0 | 4.95e-1 | 10 | 1.35e-4 | 182 |
| `icnn` | 1.72e-3 [1.33e-3, 4.14e-3], 10/10 | 1.80e-3 | 0 | 4.97e-1 | 10 | 1.37e-4 | 221 |
| `icnn_quadratic` | 1.82e-3 [9.09e-4, 3.78e-3], 10/10 | 1.69e-3 | 0 | 4.97e-1 | 10 | 1.37e-4 | 231 |

Every T12 run stopped at its step budget; no seed of any method failed on any problem (no `"diverged"` stop, no non-finite error).
"Seeds on the concave solution" counts the seeds whose max distance to the concave solution is below the threshold.
"Seeds with `max |det D^2 u - f| < min f`" counts the seeds that pass the residual half of the note's Corollary 5.1 on the shared grid: a statement about grid points, not a proof, since the supremum between them is not bounded.

## Finding 1: on P1 the QS form starts at the exact solution

`alpha` starts at `f(x0)^(1/(2d)) / sqrt(2)` and `b`, `c` and the output layer of the matrix network at 0 (the note's Section 6), so the QS form starts from `u = f(x0)^(1/d) |x - x0|^2 / 2`.
On P1, `f = 1`, `x0 = 0` and the solution is `|x|^2 / 2`: the start is the solution, the first loss is 6.7e-28 and every seed stops `converged` before its first step, with a max error of 2.2e-16.
The same holds for every problem whose solution is `f(x0)^(1/d) |x - x0|^2 / 2`, so P1 says nothing about how QS-PINN trains; `tests/test_qs_pinn.py` (`test_the_form_starts_from_alpha_squared_times_the_distance_squared`) pins the start.
T8 differs from P1 by the constant `-1/2` alone, which `c` must learn: there QS-PINN reaches 4.65e-8 [8.45e-9, 3.93e-7] in 223 to 1099 BFGS steps (53 s in the median), against the ICNN's 6.61e-6 [2.88e-6, 1.22e-5] in about 870 s, and every seed is convex on the grid and passes the residual half of the certificate.
Two seeds (1 and 2) stop `stalled`, SciPy's line search failing at errors of 6.6e-7 and 6.9e-8; `summarize` counts them as normal seeds.
T12 (Finding 3) is the problem on which the QS form starts away from the solution.

## Finding 2: the plain MLP does not select the convex branch, and its spurious stationary point is `u = g`

Under the same loss, points and optimizer as the ICNN, the plain tanh MLP succeeds in 2 seeds of 10 on P1 and 1 on T8:

- **Concave branch.** Seed 7 reaches the concave solution on both problems (max distance to it 1.05e-4 on P1 and 1.11e-4 on T8, Hessian negative definite at every grid point): the note's Example 1.1, now in the shared protocol. It also passes the residual half of the certificate (`det D^2 u > 0` at every grid point), which shows why Corollary 5.1 needs the Hessian to be positive definite at one point, the half the QS form gives by construction.
- **`u` constant.** Seeds 0, 2 and 8 on P1, and 0, 2 and 8 on T8, stop `converged` within 8 to 30 steps at a function with max error 0.500 and loss 3.4e3: `u` equals `g` to 4e-5 on the boundary and is nearly flat inside, so `det D^2 u` is near 0 and the residual near `-1` everywhere. The gradient of `det` in the Hessian is its cofactor matrix, which vanishes at a zero Hessian, so `u = g` is a stationary point of the loss: a spurious critical point of the kind the note cites (Section 2), which the QS form cannot reach, since its Hessian at `x0` is `2 A^T A` with `A` invertible.
- **The rest** stall at errors of 0.44 to 2.7, non-convex at 16% to 95% of the grid, with smallest Hessian eigenvalues down to `-1.6e3`.

Seeds 3 and 4 on P1 and 6 on T8 reach the convex solution (6.9e-6 to 9.0e-5).
The note's demonstration (Section 1: a 2-32-32-1 tanh MLP, Adam, a mean loss with boundary weight 10, points resampled every step) put all ten seeds on the concave solution; under this protocol one seed of ten goes there and most stop elsewhere, so the protocol, not only the network, decides which wrong answer the plain MLP gives.

## Finding 3: on T12 the four networks reach the same error, the level the sampling leaves

QS-PINN starts on T12 from `u = |x|^2 / 2`, whose map is the identity, and needs no fit to it; the ICNN and the plain MLP start from a fitted identity map, so all four methods start from about the same map.
After 2000 steps their medians lie within a factor 1.3 of each other (max error 1.52e-3 to 1.82e-3, map error 1.50e-3 to 1.90e-3), with overlapping bands, and their final losses within 3% (1.33e-4 to 1.37e-4).
That loss is the level the exact map itself leaves at independently drawn boundary and target points (1.1e-4 to 1.7e-4 for seeds 0 to 9, `benchmarks/icnn.md`, Finding 3): under this sampling the transport term, not the network, sets the error, and T12 does not separate the methods.
The plain MLP, not convex by construction, is convex on the grid in every seed: the problem's solution is a quadratic with Hessian `diag(2, 1/2)`, far from any non-convex function, and the identity start is convex too.

## Finding 4: the QS form is not convex by construction away from `x0`

On T12, QS-PINN's seed 1 has a negative smallest Hessian eigenvalue, `-3.93e-2`, at 1 point of the 31397 of the shared grid (fraction 3.2e-5), and its largest residual on the grid is 1.05 against `min f = 1`, so it fails the residual certificate, though its errors (max 1.92e-3, map 1.70e-3) are in the band of the others.
The other nine seeds have `eigmin` 0.497 to 0.499, the exact solution's 0.5, and pass the certificate on the grid, as every seed of the ICNNs does.
The form guarantees the structure at `x0` (Proposition 3.3), not convexity elsewhere: only the note's Theorem 4.3 or Corollary 5.1 does, and neither holds for seed 1.
The comparison table shows the same away from the note's problems: on T2, whose gradient blows up at a corner, `DirichletQSPINN` is non-convex at 13% of the grid in the median, with errors of 0.53 to 28, and on T10 (4D) at up to 2.4% (README, Comparison).

## Conventions

- **No level.** There is no published number to place in a band, so ADR 0002's levels do not apply; the threshold for "seeds below" is the comparison table's (1% of the exact oscillation on the shared grid), not a verdict.
- **The loss.** The note's loss (Section 6) is a mean of the squared residual and of the squared boundary misfit; the Dirichlet runs use the ICNN preset's loss, the residual summed over the 3400 interior and boundary points plus 100 times the mean squared misfit, so that the comparison isolates the network (`docs/methods/qs_pinn.md`). Whether the ICNN's sum or a mean is the right reading of its paper is open (#13); the QS presets follow the current code.
- **`alpha` of the ICNN plus a quadratic.** 0.25, half the smallest exact Hessian eigenvalue of T8 (1) and T12 (1/2), so the exact solutions stay representable; we chose it knowing them, an advantage a user solving an unknown problem does not have.
- **The collocation points** are drawn from the seed by each sampler and fingerprinted, not stored (#52); the methods that share a sampler share the points.
- **Device.** One CPU thread per seed, every method, as in the comparison table; `icnn_t12.csv` trained `TransportICNN` on a GPU and differs from the `icnn` rows of T12 here by rounding (`benchmarks/comparison.md`).
- **Not measured:** the note's derivative-bound certificate (Theorem 4.3, which needs bounds on the derivatives of `A`), its Lipschitz regulariser, its ablations, and the error bound of its Proposition 2.1.

## Rerun

```sh
just push
just remote-bg qs-pinn python benchmarks/run_qs_pinn.py --workers 8 \
    --out /jczhang02/maesolvers-data/data/qs_pinn.csv
# when the log ends with "wrote ...":
scp a100server:/jczhang02/maesolvers-data/data/qs_pinn.csv benchmarks/
python benchmarks/run_qs_pinn.py --summary
```

Do not `just push` while it runs: the server checkout moves, and the later runs carry the new commit.
