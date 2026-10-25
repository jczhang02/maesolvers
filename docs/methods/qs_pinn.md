# QS-PINN

QS-PINN writes the solution of $\det D^2 u = f$ as a quadratic form whose matrix comes from a plain network, so that the solution lies above its tangent plane at a reference point and its Hessian there is positive definite whatever the training does.
It is JC Zhang's own method, from an unpublished note; there is no source paper and no published table, so its benchmark is a comparison with the baselines the note names, not a reproduction (ADR 0002 gives it no level).
maesolvers ships the network part `QuadraticForm` and two presets: `DirichletQSPINN`, the input convex network's Dirichlet preset with `QuadraticForm` as its network, and `TransportQSPINN`, the same for the transport preset.
The note's ICNN baseline with a positive Hessian bound is the option `ICNN(alpha=...)`.
Comparison: [`benchmarks/qs_pinn.md`](../../benchmarks/qs_pinn.md).

## The method

The QS form (the note's Definition 3.1) with a reference point $x_0$ inside the domain is

$$
u(x) = (x - x_0)^T A(x)^T A(x) (x - x_0) + b \cdot x + c
$$

with $A(x)$ a $d \times d$ matrix from a network and $b$, $c$ trained constants.
The quadratic term is $\lvert A(x) (x - x_0) \rvert^2 \ge 0$, so $u$ lies above the plane $c + b \cdot x$ for every value of the parameters; at $x_0$ the gradient is $b$ and the Hessian is $2 A(x_0)^T A(x_0)$ (Proposition 3.3), so a concave solution, which lies below its tangent plane, cannot be represented.
The form loses no generality: every smooth uniformly convex function is one (Lemma 3.2), with $A = M^{1/2}$ and $M(x) = \int_0^1 (1 - t) D^2 u(x_0 + t (x - x_0)) dt$.

Away from $x_0$ the Hessian has two more terms, from the first and second derivatives of $A$ (Lemma 4.2), whose signs are free, so the form is not convex by construction.
The note gives two sufficient conditions.
Theorem 4.3: $u$ is uniformly convex when the variation of $A$ over the domain is small next to its smallest singular value; Example 4.5 shows the condition fails for the natural $A$ of $\sqrt{1 + \lvert x \rvert^2} - 1$, so it is conservative.
Corollary 5.1: if $\sup \lvert \det D^2 u - f \rvert \lt \inf f$ on a convex domain and $A(x_0)$ is invertible, then $\det D^2 u \gt 0$ everywhere and the Hessian, positive definite at $x_0$, stays so on the whole domain (the branch lemma, Lemma 1.2), in any dimension.
`maesolvers` checks neither at training time; `benchmarks/run_qs_pinn.py` records both sides of the residual certificate on the shared evaluation grid, an observation on grid points, not a proof.

`QuadraticForm` builds $A$ as the note's Section 6 suggests, an offset plus a smooth norm constraint (eqs. (9) and (11)):

$$
A(x) = \alpha (I + B(x)), \qquad B = \frac{\mathrm{ratio}\thinspace Z}{\sqrt{1 + \lVert Z \rVert_F^2}}
$$

where $Z(x)$ is the $d \times d$ output of a plain network (`width` units in each of `depth` hidden layers, tanh by default, as the note suggests) and $\alpha = \alpha_{\mathrm{start}} \exp (s)$, with $s$ a trained scalar that starts at $0$.
The scaling is smooth and maps every matrix into the open Frobenius ball of radius `ratio`, so $\lvert B \rvert_{\mathrm{op}} \le \lvert B \rvert_F \lt \mathrm{ratio}$.
The note's bound $\epsilon$ on $\lvert B \rvert$ is here $\mathrm{ratio}\thinspace\alpha$, tied to $\alpha$, and the note's eq. (9) needs $\epsilon \lt \alpha$, so $\alpha \gt 0$.
The exponential keeps $\alpha$ positive for every value of $s$; with $\mathrm{ratio} \lt 1$, $A$ is then invertible at every point, with singular values in $\alpha (1 - \mathrm{ratio}, 1 + \mathrm{ratio})$ (Weyl), and the condition on $A(x_0)$ in Corollary 5.1 holds by construction.
A bare trained $\alpha$ would not do: $u$ depends on $\alpha$ through $\alpha^2$ only, so at $\alpha = 0$, where $A = 0$ and $u = b \cdot x + c$ has a zero Hessian, every gradient through $A$ vanishes, and with the plane fitted to the boundary data it is a stationary point of the loss, one the training could stop at.
`Result.info["network"]["alpha"]` records the final $\alpha$, so a run that drifts towards $0$ is visible.
Proposition 4.4 (a) says the scale of $A$ does not matter for the convexity condition, only its relative variation; `ratio` bounds that variation, $\lvert A / \alpha - I \rvert_F \lt \mathrm{ratio}$, so the singular values of $A$ at a point differ by a factor below $(1 + \mathrm{ratio}) / (1 - \mathrm{ratio})$, at the price of a smaller margin $\sigma_{\min}(A) \ge \alpha (1 - \mathrm{ratio})$.
$\alpha$ starts at $\alpha_{\mathrm{start}} = f(x_0)^{1/(2d)} / \sqrt{2}$ (Section 6), $b$, $c$ at $0$, and the output layer of the plain network at $0$, so $B = 0$ at the start and the training starts from $u = \lvert x - x_0 \rvert^2 f(x_0)^{1/d} / 2$: its Hessian determinant is $f(x_0)$ everywhere and its gradient map is $f(x_0)^{1/d} (x - x_0)$, the identity map when $f(x_0) = 1$ and $x_0 = 0$.
The other weights and biases of the plain network start from the standard normal distribution, as `MLP`'s.

The presets keep every other part of the input convex network's presets (`docs/methods/icnn.md`), so that with the same seed QS-PINN and the ICNN train on the same collocation points with the same loss and optimizer and differ in the network alone:

- `DirichletQSPINN()` is `NeuralSolver(network=QuadraticForm(width=10, depth=5), loss=PenaltyLoss(convexity_weight=0, boundary_weight=100, equation_at_boundary=True), sampler=UniformSampler(interior_points=3000, boundary_points=400), optimizer=BFGS(), max_steps=10_000)`;
- `TransportQSPINN()` is `NeuralSolver(network=QuadraticForm(width=10, depth=4), loss=HausdorffLoss(boundary_weight=1), sampler=TransportSampler(800, 800, 800), optimizer=Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200), max_steps=2000, tol=1e-7)`.

`TransportICNN` fits its network to $\nabla u = x$ before the training (`IdentityStart`); `TransportQSPINN` needs no fit, since its start already has the gradient map $x - x_0$ when $f(x_0) = 1$.
That is the identity map only when $x_0 = 0$, as on T12, whose domain is the unit disk: on a domain centred elsewhere the two presets start from different maps.
Both presets take `reference=` to move $x_0$.
The note's loss is a mean of the squared residual and of the squared boundary misfit; the preset takes the ICNN's loss instead, the residual summed over the interior and boundary points plus $100$ times the mean squared misfit, so that the comparison isolates the network.
`QuadraticForm` composes with any loss, sampler and optimizer: `docs/guide.md` shows how to swap a part.
ADR 0004 lists QS among the losses (`a loss (penalty, QS)`); the QS form is a network part here, since it is the representation of $u$, not an objective.

The ICNN baseline with a positive Hessian bound is `ICNN(alpha=a)`: the input convex network plus $a \lvert x \rvert^2 / 2$, whose Hessian is at least $a I$ (the note's Section 2).
$a$ must not exceed the smallest Hessian eigenvalue of the solution, or the network cannot represent it, so it needs the scale of the Hessian in advance.

## Source paper

None published.
The method is in JC Zhang, "二次型结构 PINN (QS-PINN)", a note of October 2026 in the research repository (`writing/notes/qs-pinn/main.tex` in jczhang02/monge-ampere-equations, private), which states that none of its claims has training support yet.
The numbers here follow its compiled version: Definition 3.1 (the form), Lemma 3.2 (generality), Proposition 3.3 (the structure at $x_0$), Theorem 4.3 (the derivative condition), Proposition 4.4 (its invariances), Example 4.5, Corollary 5.1 (the residual certificate), Section 6 (eqs. (9) to (11), the offset, $\alpha$'s start and the smooth scaling), Section 7 (Table 2, the problems P1 to P3, and the baselines).
The baselines are those of `docs/methods/penalty_mlp.md` and `docs/methods/icnn.md`.

## Parameters

| Part | Parameter | Default | Where it comes from |
|---|---|---|---|
| `QuadraticForm` | `width`, `depth` | $10$, $5$ (`DirichletQSPINN`), $10$, $4$ (`TransportQSPINN`) | not in the note; the sizes of the ICNN of the preset it replaces, giving $518$ and $408$ trained parameters in 2D against the ICNN's $563$ and $433$ |
| `QuadraticForm` | `activation` | `"tanh"` | the note, Section 6 (smooth, with bounded first and second derivatives) |
| `QuadraticForm` | `ratio` | $0.6$ | ours, below |
| `QuadraticForm`, `DirichletQSPINN`, `TransportQSPINN` | `reference` | None, the centre of the bounding box | the note, Definition 3.1 ("for example the centre of the domain") |
| `QuadraticForm` | the start of `alpha` | $f(x_0)^{1/(2d)} / \sqrt{2}$ | the note, Section 6 |
| `DirichletQSPINN` | `boundary_weight`, `interior_points`, `boundary_points`, `max_steps` | $100$, $3000$, $400$, $10000$ | `DirichletICNN`'s |
| `TransportQSPINN` | `boundary_weight`, the points, `adam_lr`, `adam_steps`, `max_steps`, `tol` | $1$, $800$ each, $10^{-3}$, $200$, $2000$, $10^{-7}$ | `TransportICNN`'s |
| `ICNN` | `alpha` | $0$ | $0$ is the plain ICNN; the benchmark takes $0.25$, half the smallest exact Hessian eigenvalue of T8 and T12 |

`ratio` $= 0.6$ holds the exact solutions of both of the note's test problems with a margin.
T8 (and P1) needs $A = I / \sqrt{2}$, which is $B = 0$ at $\alpha$'s start, for any `ratio`.
T12, $x_1^2 + 3.5 x_1 + x_2^2 / 4$ with $x_0 = 0$, needs $A^T A = \operatorname{diag}(1, 1/4)$: $A = \operatorname{diag}(1, 1/2)$ up to an orthogonal factor, so $(I + B)$ must have singular values in the ratio $2$.
The smallest $\lvert B \rvert_F$ that does it is $0.447$, at $\alpha = 5/6$ with $B = \operatorname{diag}(1/5, -2/5)$; at $\alpha$'s start, $1 / \sqrt{2}$, it is $\lvert \operatorname{diag}(\sqrt{2} - 1, 1/\sqrt{2} - 1) \rvert_F = 0.507$.
So `ratio` must exceed $0.447$ for T12 (the note's Frobenius figure, $\epsilon \gt 0.35$ at $\alpha = 0.75$, is the ratio $0.47$ at that $\alpha$); $0.6$ leaves $\lvert Z \rvert_F = 1.1$ at the best $\alpha$, away from the saturation of the scaling, and bounds the condition number of $A^T A$ at a point by $16$.
The value comes from T12's exact solution and from 2D arithmetic; the note gives none.
The same arithmetic on $\exp (\lvert x \rvert^2 / 2)$ on $(0, 1)^d$ with $x_0$ the centre, whose canonical $A = M^{1/2}$ varies over the domain, gives a best $\alpha$ with largest $\lvert A / \alpha - I \rvert_F$ of $0.28$ in 2D (T1), $0.46$ in 3D (T9) and $0.64$ in 4D (T10), growing with $d$ as the note's Section 6 warns ($\lvert B \rvert_F$ up to $\sqrt{d}$ times $\lvert B \rvert_{\mathrm{op}}$); `python benchmarks/run_qs_pinn.py --canonical` prints these.
So $0.6$ holds the canonical representation in 2D and 3D, not in 4D; $A$ is not unique (only $\lvert A (x - x_0) \rvert$ enters $u$), so another representation may fit inside $0.6$ on T10, and the training there stalls before this bound matters (`benchmarks/qs_pinn.md`, Finding 5).

## Known limits

- The form is not convex by construction away from $x_0$: on T2 the median seed is non-convex at $18.7$% of the grid, on T10 five seeds of ten are non-convex at some grid points, and on T12 the first version, with a bare $\alpha$, ended one seed of ten with a negative Hessian eigenvalue at a grid point (`benchmarks/qs_pinn.md`, Finding 4); only the sufficient conditions of the note certify convexity, and `maesolvers` checks neither (the benchmark observes the residual one on a grid).
- When $f$ is constant and the solution is $f(x_0)^{1/d} \lvert x - x_0 \rvert^2 / 2$, as on the note's P1, the start is the solution and the training stops before its first step (`benchmarks/qs_pinn.md`, Finding 1); such a problem tests the representation, not the training.
- $f(x_0)$ must be positive: the form gives $\det D^2 u(x_0) = 2^d \alpha^{2d} \det (I + B)^2 \gt 0$, so it cannot meet $f(x_0) = 0$, and `QuadraticForm.supports` is False there (`QuadraticForm.refusal` names the reason). With the default reference, the presets do not support T4, T20 and T20c, whose $f$ vanishes on a disk around the centre of their box, and their error says so; `DirichletQSPINN(reference=(0.1, 0.1))` does.
  The error states the reason at $x_0$ as a fact and does not promise that another point would do: on T39 and T40 the default reference fails too ($f(0, 0) = 0$), but the sampler refuses the `DiskUnionSquare` domain wherever $x_0$ lies (#74).
- $\alpha$ stays positive, but it can drift towards $0$ ($s$ towards minus infinity), where the margin $\sigma_{\min}(A) \ge \alpha (1 - \mathrm{ratio})$ vanishes and the training approaches the flat stationary point $u = b \cdot x + c$; `Result.info["network"]["alpha"]` shows the final value next to $\alpha_{\mathrm{start}}$.
- `ratio` bounds $\lvert A / \alpha - I \rvert_F$ at every point, with one $\alpha$ for the whole domain: a solution whose $M$ (above) varies more than that over the domain, such as one whose Hessian blows up at a corner (T2), has no canonical representation inside the bound. The binding constraint is the Frobenius ball, not the spread of the singular values at a point (below $4$ at $0.6$), and it grows with the dimension: on $\exp (\lvert x \rvert^2 / 2)$ the canonical $A$ needs `ratio` above $0.28$ in 2D, $0.46$ in 3D and $0.64$ in 4D (Parameters), so the default $0.6$ is justified for 2D and 3D only. On T10 this does not decide the error: with `ratio` $= 0.9$ every seed still stalls, at a max error of $9.3 \times 10^{-1}$ in the median against $1.1$ at $0.6$ (`benchmarks/qs_pinn.md`, Finding 5).
- The reference point is where the form is anchored; the note leaves open whether a solution whose minimum lies far from $x_0$ (T12, minimum at $x_1 = -7/4$, outside the disk) trains as well (its Q5).
- The note's derivative-bound certificate (Theorem 4.3) and its Lipschitz regulariser $\sum_k \lVert \partial_k A \rVert_F^2$ are not implemented, nor are the ablations its Section 7 lists (fixed $\alpha$, hard projection, Adam against L-BFGS, dimensions $2$ to $4$).
- The training is deterministic only on one machine with one torch thread count, as for every neural preset.

## Comparison

No ADR 0002 level: the note publishes no numbers, so [`benchmarks/qs_pinn.md`](../../benchmarks/qs_pinn.md) compares QS-PINN with the plain MLP, the penalty MLP, the ICNN, the ICNN plus $0.25 \lvert x \rvert^2 / 2$ and, on T12, the transport ICNN, over seeds $0$ to $9$ on the note's P1 (T8 with $g = 1/2$), the catalogue's T8 and T12, and runs QS-PINN alone on T10 with `ratio` $0.6$ and $0.9$.
Max error on the shared grid, median $[5, 95]$% over the seeds:

| Problem | QS-PINN | Plain MLP | Penalty MLP | ICNN | ICNN plus $0.25 \lvert x \rvert^2 / 2$ |
|---|---|---|---|---|---|
| P1 | $2.22 \times 10^{-16}$ $[2.22 \times 10^{-16}, 2.22 \times 10^{-16}]$ (the start, Finding 1) | $6.31 \times 10^{-1}$ $[3.46 \times 10^{-5}, 2.47 \times 10^{0}]$ | $5.26 \times 10^{-3}$ $[4.10 \times 10^{-3}, 9.37 \times 10^{-3}]$ | $7.72 \times 10^{-6}$ $[3.74 \times 10^{-6}, 1.34 \times 10^{-5}]$ | $4.96 \times 10^{-6}$ $[3.50 \times 10^{-6}, 1.11 \times 10^{-5}]$ |
| T8 | $1.72 \times 10^{-7}$ $[4.03 \times 10^{-8}, 3.82 \times 10^{-7}]$ | $5.28 \times 10^{-1}$ $[1.97 \times 10^{-1}, 1.94 \times 10^{0}]$ | $5.04 \times 10^{-3}$ $[3.90 \times 10^{-3}, 8.96 \times 10^{-3}]$ | $6.61 \times 10^{-6}$ $[2.88 \times 10^{-6}, 1.22 \times 10^{-5}]$ | $7.31 \times 10^{-6}$ $[4.09 \times 10^{-6}, 1.18 \times 10^{-5}]$ |
| T12 | $1.32 \times 10^{-3}$ $[8.80 \times 10^{-4}, 3.81 \times 10^{-3}]$ | $1.82 \times 10^{-3}$ $[9.07 \times 10^{-4}, 3.49 \times 10^{-3}]$ | not run | $1.72 \times 10^{-3}$ $[1.33 \times 10^{-3}, 4.14 \times 10^{-3}]$ (`TransportICNN`) | $1.82 \times 10^{-3}$ $[9.09 \times 10^{-4}, 3.78 \times 10^{-3}]$ |

On T8 QS-PINN is about $38$ times more accurate than the ICNN in the median and $18$ times faster; on T12 the four networks reach the error the sampling of the transport term leaves (Finding 3); the plain MLP reaches the concave solution in one seed of ten and a flat spurious stationary point in three (Finding 2).
On the rest of the shared test set (README, Comparison) `DirichletQSPINN` is near `DirichletICNN` on T1 and T9 (where all ten seeds meet the threshold, against seven), worse on T2 (one seed diverges), fails the threshold in every seed on T10 as the ICNN does (max error about $1$ against about $900$; Finding 5), and refuses T20 ($f(x_0) \le 0$; T4, outside the shared set, likewise).
