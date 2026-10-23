# QS-PINN

QS-PINN writes the solution of `det D^2 u = f` as a quadratic form whose matrix comes from a plain network, so that the solution lies above its tangent plane at a reference point and its Hessian there is positive definite whatever the training does.
It is JC Zhang's own method, from an unpublished note; there is no source paper and no published table, so its benchmark is a comparison with the baselines the note names, not a reproduction (ADR 0002 gives it no level).
maesolvers ships the network part `QuadraticForm` and two presets: `DirichletQSPINN`, the input convex network's Dirichlet preset with `QuadraticForm` as its network, and `TransportQSPINN`, the same for the transport preset.
The note's ICNN baseline with a positive Hessian bound is the option `ICNN(alpha=...)`.
Comparison: [`benchmarks/qs_pinn.md`](../../benchmarks/qs_pinn.md).

## The method

The QS form (the note's Definition 3.1) with a reference point `x0` inside the domain is

```text
u(x) = (x - x0)^T A(x)^T A(x) (x - x0) + b . x + c
```

with `A(x)` a `d x d` matrix from a network and `b`, `c` trained constants.
The quadratic term is `|A(x) (x - x0)|^2 >= 0`, so `u` lies above the plane `c + b . x` for every value of the parameters; at `x0` the gradient is `b` and the Hessian is `2 A(x0)^T A(x0)` (Proposition 3.3), so a concave solution, which lies below its tangent plane, cannot be represented.
The form loses no generality: every smooth uniformly convex function is one (Lemma 3.2), with `A = M^(1/2)` and `M(x) = int_0^1 (1 - t) D^2 u(x0 + t (x - x0)) dt`.

Away from `x0` the Hessian has two more terms, from the first and second derivatives of `A` (Lemma 4.2), whose signs are free, so the form is not convex by construction.
The note gives two sufficient conditions.
Theorem 4.3: `u` is uniformly convex when the variation of `A` over the domain is small next to its smallest singular value; Example 4.5 shows the condition fails for the natural `A` of `sqrt(1 + |x|^2) - 1`, so it is conservative.
Corollary 5.1: if `sup |det D^2 u - f| < inf f` on a convex domain and `A(x0)` is invertible, then `det D^2 u > 0` everywhere and the Hessian, positive definite at `x0`, stays so on the whole domain (the branch lemma, Lemma 1.2), in any dimension.
`maesolvers` checks neither at training time; `benchmarks/run_qs_pinn.py` records both sides of the residual certificate on the shared evaluation grid, an observation on grid points, not a proof.

`QuadraticForm` builds `A` as the note's Section 6 suggests, an offset plus a smooth norm constraint (eqs. (9) and (11)):

```text
A(x) = alpha (I + B(x)),   B = ratio Z / sqrt(1 + |Z|_F^2)
```

where `Z(x)` is the `d x d` output of a plain network (`width` units in each of `depth` hidden layers, tanh by default, as the note suggests) and `alpha` is a trained scalar.
The scaling is smooth and maps every matrix into the open Frobenius ball of radius `ratio`, so `|B|_op <= |B|_F < ratio`.
The note's bound `epsilon` on `|B|` is here `ratio |alpha|`, tied to `alpha`: with `ratio < 1` it stays below `|alpha|` for every trained `alpha`, so `A` is invertible at every point, with singular values in `|alpha| (1 - ratio, 1 + ratio)` (Weyl), and the condition on `A(x0)` in Corollary 5.1 holds by construction.
Proposition 4.4 (a) says the scale of `A` does not matter for the convexity condition, only its relative variation; `ratio` sets that trade directly: the singular values of `A` at a point differ by a factor below `(1 + ratio) / (1 - ratio)`, at the price of a smaller margin `sigma_min(A) >= |alpha| (1 - ratio)`.
`alpha` starts at `f(x0)^(1/(2d)) / sqrt(2)` (Section 6), `b`, `c` at 0, and the output layer of the plain network at 0, so `B = 0` at the start and the training starts from `u = |x - x0|^2 f(x0)^(1/d) / 2`: its Hessian determinant is `f(x0)` everywhere and its gradient map is `f(x0)^(1/d) (x - x0)`, the identity map when `f(x0) = 1` and `x0 = 0`.
The other weights and biases of the plain network start from the standard normal distribution, as `MLP`'s.

The presets keep every other part of the input convex network's presets (`docs/methods/icnn.md`), so that with the same seed QS-PINN and the ICNN train on the same collocation points with the same loss and optimizer and differ in the network alone:

- `DirichletQSPINN()` is `NeuralSolver(network=QuadraticForm(width=10, depth=5), loss=PenaltyLoss(convexity_weight=0, boundary_weight=100, equation_at_boundary=True), sampler=UniformSampler(interior_points=3000, boundary_points=400), optimizer=BFGS(), max_steps=10_000)`;
- `TransportQSPINN()` is `NeuralSolver(network=QuadraticForm(width=10, depth=4), loss=HausdorffLoss(boundary_weight=1), sampler=TransportSampler(800, 800, 800), optimizer=Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200), max_steps=2000, tol=1e-7)`.

`TransportICNN` fits its network to `grad u = x` before the training (`IdentityStart`); `TransportQSPINN` needs no fit, since its start already has the gradient map `x - x0` when `f(x0) = 1`, as on T12.
The note's loss is a mean of the squared residual and of the squared boundary misfit; the preset takes the ICNN's loss instead, the residual summed over the interior and boundary points plus 100 times the mean squared misfit, so that the comparison isolates the network.
`QuadraticForm` composes with any loss, sampler and optimizer: `docs/guide.md` shows how to swap a part.

The ICNN baseline with a positive Hessian bound is `ICNN(alpha=a)`: the input convex network plus `a |x|^2 / 2`, whose Hessian is at least `a I` (the note's Section 2).
`a` must not exceed the smallest Hessian eigenvalue of the solution, or the network cannot represent it, so it needs the scale of the Hessian in advance.

## Source paper

None published.
The method is in JC Zhang, "二次型结构 PINN (QS-PINN)", a note of October 2026 in the research repository (`writing/notes/qs-pinn/main.tex` in jczhang02/monge-ampere-equations, private), which states that none of its claims has training support yet.
The numbers here follow its compiled version: Definition 3.1 (the form), Lemma 3.2 (generality), Proposition 3.3 (the structure at `x0`), Theorem 4.3 (the derivative condition), Proposition 4.4 (its invariances), Example 4.5, Corollary 5.1 (the residual certificate), Section 6 (eqs. (9) to (11), the offset, `alpha`'s start and the smooth scaling), Section 7 (Table 2, the problems P1 to P3, and the baselines).
The baselines are those of `docs/methods/penalty_mlp.md` and `docs/methods/icnn.md`.

## Parameters

| Part | Parameter | Default | Where it comes from |
|---|---|---|---|
| `QuadraticForm` | `width`, `depth` | 10, 5 (`DirichletQSPINN`), 10, 4 (`TransportQSPINN`) | not in the note; the sizes of the ICNN of the preset it replaces, giving 518 and 408 trained parameters in 2D against the ICNN's 563 and 433 |
| `QuadraticForm` | `activation` | `"tanh"` | the note, Section 6 (smooth, with bounded first and second derivatives) |
| `QuadraticForm` | `ratio` | 0.6 | ours, below |
| `QuadraticForm` | `reference` | None, the centre of the bounding box | the note, Definition 3.1 ("for example the centre of the domain") |
| `QuadraticForm` | the start of `alpha` | `f(x0)^(1/(2d)) / sqrt(2)` | the note, Section 6 |
| `DirichletQSPINN` | `boundary_weight`, `interior_points`, `boundary_points`, `max_steps` | 100, 3000, 400, 10000 | `DirichletICNN`'s |
| `TransportQSPINN` | `boundary_weight`, the points, `adam_lr`, `adam_steps`, `max_steps`, `tol` | 1, 800 each, 1e-3, 200, 2000, 1e-7 | `TransportICNN`'s |
| `ICNN` | `alpha` | 0 | 0 is the plain ICNN; the benchmark takes 0.25, half the smallest exact Hessian eigenvalue of T8 and T12 |

`ratio = 0.6` holds the exact solutions of both of the note's test problems with a margin.
T8 (and P1) needs `A = I / sqrt(2)`, which is `B = 0` at `alpha`'s start, for any `ratio`.
T12, `x1^2 + 3.5 x1 + x2^2 / 4` with `x0 = 0`, needs `A^T A = diag(1, 1/4)`: `A = diag(1, 1/2)` up to an orthogonal factor, so `(I + B)` must have singular values in the ratio 2.
The smallest `|B|_F` that does it is `0.447`, at `alpha = 5/6` with `B = diag(1/5, -2/5)`; at `alpha`'s start, `1 / sqrt(2)`, it is `|diag(sqrt(2) - 1, 1/sqrt(2) - 1)|_F = 0.507`.
So `ratio` must exceed 0.447 for T12 (the note's Frobenius figure, `epsilon > 0.35` at `alpha = 0.75`, is the ratio 0.47 at that `alpha`); 0.6 leaves `|Z|_F = 1.1` at the best `alpha`, away from the saturation of the scaling, and bounds the condition number of `A^T A` at a point by 16.

## Known limits

- The form is not convex by construction away from `x0`: on T12 one seed of ten ends with a negative Hessian eigenvalue at a grid point, and on T2 the median seed is non-convex at 13% of the grid (`benchmarks/qs_pinn.md`, Finding 4); only the sufficient conditions of the note certify convexity, and `maesolvers` checks neither (the benchmark observes the residual one on a grid).
- When `f` is constant and the solution is `f(x0)^(1/d) |x - x0|^2 / 2`, as on the note's P1, the start is the solution and the training stops before its first step (`benchmarks/qs_pinn.md`, Finding 1); such a problem tests the representation, not the training.
- `f(x0)` must be positive: the form gives `det D^2 u(x0) = 2^d alpha^(2d) det(I + B)^2 > 0`, so it cannot meet `f(x0) = 0`, and `QuadraticForm.supports` is False there. With the default reference, the preset does not support T20, whose `f` vanishes on a disk around the centre of its box; `QuadraticForm(reference=(0.1, 0.1))` composed by hand does.
- `ratio` bounds how much the singular values of `A` can differ at a point (a factor below 4 at 0.6), and `alpha` is one number for the whole domain: a solution whose `M` (above) varies by more than that over the domain, such as one whose Hessian blows up at a corner (T2), cannot be represented exactly.
- The reference point is where the form is anchored; the note leaves open whether a solution whose minimum lies far from `x0` (T12, minimum at `x1 = -7/4`, outside the disk) trains as well (its Q5).
- The note's derivative-bound certificate (Theorem 4.3) and its Lipschitz regulariser `sum_k |d_k A|_F^2` are not implemented, nor are the ablations its Section 7 lists (fixed `alpha`, hard projection, Adam against L-BFGS, dimensions 2 to 4).
- The training is deterministic only on one machine with one torch thread count, as for every neural preset.

## Comparison

No ADR 0002 level: the note publishes no numbers, so [`benchmarks/qs_pinn.md`](../../benchmarks/qs_pinn.md) compares QS-PINN with the plain MLP, the penalty MLP, the ICNN, the ICNN plus `0.25 |x|^2 / 2` and, on T12, the transport ICNN, over seeds 0 to 9 on the note's P1 (T8 with `g = 1/2`), the catalogue's T8 and T12.
Max error on the shared grid, median [5%, 95%] over the seeds:

| Problem | QS-PINN | Plain MLP | Penalty MLP | ICNN | ICNN plus `0.25 |x|^2 / 2` |
|---|---|---|---|---|---|
| P1 | 2.22e-16 [2.22e-16, 2.22e-16] (the start, Finding 1) | 6.31e-1 [3.46e-5, 2.47e0] | 5.26e-3 [4.10e-3, 9.37e-3] | 7.72e-6 [3.74e-6, 1.34e-5] | 4.96e-6 [3.50e-6, 1.11e-5] |
| T8 | 4.65e-8 [8.45e-9, 3.93e-7] | 5.28e-1 [1.97e-1, 1.94e0] | 5.04e-3 [3.90e-3, 8.96e-3] | 6.61e-6 [2.88e-6, 1.22e-5] | 7.31e-6 [4.09e-6, 1.18e-5] |
| T12 | 1.52e-3 [1.01e-3, 3.92e-3] | 1.82e-3 [9.07e-4, 3.49e-3] | not run | 1.72e-3 [1.33e-3, 4.14e-3] (`TransportICNN`) | 1.82e-3 [9.09e-4, 3.78e-3] |

On T8 QS-PINN is about 140 times more accurate than the ICNN in the median and 16 times faster; on T12 the four networks reach the error the sampling of the transport term leaves (Finding 3); the plain MLP reaches the concave solution in one seed of ten and a flat spurious stationary point in three (Finding 2).
On the rest of the shared test set (README, Comparison) `DirichletQSPINN` is near `DirichletICNN` on T1 and T9 (where all ten seeds meet the threshold, against seven), worse on T2, fails the threshold in every seed on T10 as the ICNN does (max error about 1 against about 900), and refuses T20 (`f(x0) = 0`).
