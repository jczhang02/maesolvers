# Input convex network (ICNN)

The input convex network of Nystrom and Vestberg (2023) solves the Dirichlet problem of $\det D^2 u = f$ with a network that is convex in its input by construction, so the solution it finds is convex whatever the training does.
The boundary condition enters the loss as a penalty.
maesolvers ships it as the network part `ICNN` and the preset `DirichletICNN`, a `NeuralSolver` composed of four parts (ADR 0004), trained on the shared loop of ADR 0008.
The same network solves the transport problem in the convex PINN of Caboussat and Peruso (2025), the preset `TransportICNN`, with a loss on the transport condition (The transport ICNN, below).
Reproduction: [`benchmarks/icnn.md`](../../benchmarks/icnn.md).

## The method

The network (the paper's eq. (3.7), after Amos, Xu and Kolter 2017) with $k$ hidden layers is

$$
\begin{aligned}
z_1 &= s(L_0 x + b_0) \cr
z_{j+1} &= s(W_j z_j + L_j x + b_j), \quad j = 1, \dots, k - 1 \cr
u(x) &= W_k z_k + L_k x + b_k
\end{aligned}
$$

where $s(t) = \log(1 + e^t)$ is the softplus, the $L_j$ and $b_j$ are free, and every $W_j$ has non-negative entries.
Each $z_j$ is convex in $x$: an affine map of $x$ is convex, a non-negative combination of convex functions is convex, and a convex non-decreasing function of a convex function is convex.
So $u$ is convex for every value of the parameters, in any dimension, and its Hessian is positive semidefinite at every point, not only at the collocation points.
The softplus is smooth, so $u$ has a Hessian everywhere; ReLU is convex but has no second derivative, and tanh and the sigmoid are not convex, so `ICNN` offers no other activation.
As in the paper (Section 4), $W_j$ is the elementwise square $V_j^2$ of the trained parameter $V_j$, and every parameter starts from the normal distribution, with standard deviation `init_std = 1`.
With five hidden layers of $10$ units in 2D the network has $563$ parameters (Example 4.2): $30$ in the first layer, $130$ in each of the four later hidden layers, $13$ in the output layer.

A convex network cannot use the ansatz $u = G + D N$ of the penalty MLP (`docs/methods/penalty_mlp.md`): the product $D N$ of convex functions need not be convex.
So the boundary condition is a penalty, and the loss (eq. (3.8), as Example 4.2 uses it) over the interior collocation points $x_1, \dots, x_M$ and the boundary points $y_1, \dots, y_K$ is

$$
E = \sum_j (\det D^2 u(x_j) - f(x_j))^2 + \sum_k (\det D^2 u(y_k) - f(y_k))^2 + C (1/K) \sum_k (u(y_k) - g(y_k))^2
$$

with $C = 100$: the equation term is a sum, not a mean, and in Example 4.2 it runs over the boundary points too.
The preset follows the print; a mean would be the sum with $C = 3.4 \times 10^5$, up to a factor $1/3400$, and our diagnostic runs with the mean meet the paper's stopping test but leave T6 a mismatch (Known limits, below).
This is `PenaltyLoss(convexity_weight=0, boundary_weight=100, equation_at_boundary=True)`.
So the ICNN differs from the penalty MLP in its loss as well as in its network: no convexity penalty, a boundary penalty instead of the ansatz, and the equation at the boundary points.
It differs in its optimizer too, below.

The collocation points are uniform random, $3000$ inside the domain and $400$ on the boundary, each piece of the boundary getting points in proportion to its length (`UniformSampler`), drawn from the seed by the sampler; whether ADR 0008's shared collocation set is per problem or per sampler, and whether it is stored, is open (#52).
The optimizer is BFGS with a dense inverse Hessian approximation, one iteration of SciPy's `minimize(method="BFGS")`, which the paper ran, per training step (`BFGS`): the approximation starts as the identity and gets the BFGS update after every step, and the step comes from SciPy's own strong Wolfe line search, so the iterates are SciPy's up to rounding.
The limited-memory `LBFGS` of the penalty MLP trains this network far less well: on T1 it leaves the loss about $50000$ times higher after $10000$ steps, and the max error about $270$ times higher (`benchmarks/icnn.md`, Finding 1).
The training stops when the largest component of the loss gradient is at most `tol = 1e-5`, the default of SciPy's BFGS, or after `max_steps = 10000` steps.
It runs on the CPU unless `accelerator` says otherwise.

## Source paper

K. Nystrom, M. Vestberg, "Solving the Dirichlet problem for the Monge-Ampere equation using neural networks", J. Comput. Math. Data Sci. 8 (2023) 100080; arXiv 2110.03310 (v3, 2023-06-13, the version read here).
Section 3.2 defines the method and Example 4.2 runs it on T1, with Figure 2 as its result; Example 4.6 runs it on T6, with Table 1.
The network construction is that of B. Amos, L. Xu, J. Z. Kolter, "Input convex neural networks", ICML 2017.
The other ICNN of the roadmap, Caboussat and Peruso (2025), trains an ICNN on the transport boundary condition: `TransportICNN`, below.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `DirichletICNN` | `width`, `depth` | $10$, $5$ | units per hidden layer and hidden layers of the network (`ICNN`) |
| `DirichletICNN` | `init_std` | $1$ | standard deviation of every initial parameter of the network (`ICNN`) |
| `ICNN` | `alpha` | $0$ | coefficient of $\alpha \lvert x \rvert^2 / 2$ added to the network, whose Hessian is then at least $\alpha I$; not in the paper, not a field of the presets: compose `NeuralSolver(network=ICNN(alpha=0.25), ...)`, the baseline of the QS-PINN note (`qs_pinn.md`) |
| `DirichletICNN` | `boundary_weight` | $100$ | $C$, the weight of the boundary misfit (`PenaltyLoss`), positive: without the ansatz, this term alone imposes $g$ |
| `DirichletICNN` | `interior_points`, `boundary_points` | $3000$, $400$ | collocation points inside and on the boundary (`UniformSampler`) |
| `NeuralSolver` | `max_steps` | $10000$ (preset), $1000$ | most training steps |
| `NeuralSolver` | `tol` | $10^{-5}$ | largest gradient component at which the training stops, converged |
| `NeuralSolver` | `seed` | $0$ | seeds the collocation points and the initial network |
| `NeuralSolver` | `accelerator` | `"cpu"` | `"cpu"`, `"gpu"`, or `"auto"` (a GPU when there is one) |
| `PenaltyLoss` | `convexity_weight`, `equation_at_boundary` | $0$, True (preset) | the penalty on negative $u_{xx}$, $u_{yy}$, which a convex network does not need, and whether the equation is also summed at the boundary points |
| `BFGS` | none | | |

The preset takes the parameters of its parts flat: `DirichletICNN(width=20, seed=3)` is `NeuralSolver(network=ICNN(width=20), loss=PenaltyLoss(convexity_weight=0.0, boundary_weight=100.0, equation_at_boundary=True), sampler=UniformSampler(), optimizer=BFGS(), max_steps=10_000, seed=3)`.
`ICNN` also composes with the other parts; `equation_at_boundary=False` gives the loss of the paper's Section 5, whose equation term runs over the interior points only.
`docs/guide.md` shows how to swap a part or write one.

## Differences from the paper

- The paper restarted with Adam when BFGS stopped on a failed line search (Example 4.1); the preset does not: a step that changes nothing ends the training (`stop = "stalled"`).
- The paper gives the iterations of one T1 run (convergence after $2287$) and caps the T6 runs of Table 1 at $10000$; the preset caps every run at $10000$ steps.
- The paper does not give the standard deviation of the initial weights; the preset uses $1$ for every parameter, the roots $V_j$ included, as the paper's "normal distribution" reads literally (TensorFlow 1's default for a normal draw; TensorFlow 2's `RandomNormal` defaults to $0.05$). Under the dense BFGS the choice matters less than the optimizer (`benchmarks/icnn.md`, Finding 1).
- The training stops when the largest gradient component is at most $10^{-5}$, SciPy's default test (`norm=inf`); the paper's text says the Euclidean norm.

## Known limits

- Dirichlet only, and only on domains that sample their boundary (`Box` in any dimension, `Disk`, `Ellipse`, `CutQuarterDisk`): `solve` raises `UnsupportedProblemError` for the transport problems T12 and T16, which `TransportICNN` solves.
- The boundary condition is a penalty, so $u$ meets it only approximately: on T1 and T6 the max error lies on the boundary in every seed.
- Within the $10000$-step budget no seed meets the stopping test, and on T6 the errors are about $13$ times those of the paper's Table 1 (`benchmarks/icnn.md`, T6).
  The printed sum explains part of it: read as a mean, the equation term weighs the boundary misfit, where the max error is, $3400$ times more.
  The mean reading (BFGS's start $H_0 = I$ on the mean form) meets the stopping test in $7$ of $10$ seeds on T6 after a few thousand steps, but stays a mismatch, about $3.6$ times above Table 1 (max $3.09 \times 10^{-5}$ $[1.21 \times 10^{-5}, 4.96 \times 10^{-5}]$ against $8.58 \times 10^{-6}$).
  Only `DirichletICNN(boundary_weight=340_000, tol=0.034)`, the mean reading with a $3400$ times larger BFGS start than SciPy's, puts Table 1 inside its band on T6 (max $5.47 \times 10^{-6}$ $[2.58 \times 10^{-6}, 1.68 \times 10^{-5}]$; `benchmarks/icnn.md`, Finding 2).
  The preset keeps the sum until #13 decides.
- Convexity holds by construction, but the class of functions is narrower than that of a plain network of the same size: every hidden unit is convex, so a solution that is nearly affine in some direction, or whose Hessian degenerates, as for $f = 0$ on a set (T4, T5, T20), is harder to represent.
- With the standard normal start, the squared roots have mean $1$, so the activations grow from layer to layer and the first loss is very large (about $6 \times 10^{19}$ on T1 for seed $0$); BFGS brings it down within a few hundred steps, but the start is far from the solution.
  The first loss grows with the dimension, since every layer reads all $d$ inputs: at the defaults and seeds $0$ to $2$ it is $6 \times 10^{19}$ to $10^{21}$ on T1 and T6 (2D), $2 \times 10^{26}$ to $3 \times 10^{28}$ on T9 (3D) and $2 \times 10^{33}$ to $3 \times 10^{36}$ on T10 (4D).
- `DirichletICNN` runs on T9 and T10 (3D and 4D Dirichlet problems), but its accuracy there is not measured: the paper's Tables 2 and 3 are not reproduced (`benchmarks/icnn.md`, Not compared), and the tests train only a few steps in 3D.
- `BFGS` stores a dense $n \times n$ matrix and multiplies by it at every step, so it suits networks of a few thousand parameters at most; for a larger `ICNN`, pass `LBFGS()` and expect slower training.
- The training is deterministic only on one machine with one torch thread count; `result.info` records `device` and `threads`, and the receipts use one thread per seed.

## The transport ICNN

`TransportICNN` is the convex PINN of Caboussat and Peruso (2025) for the second boundary value problem: $\nabla u$ maps the domain $X$ onto the target set $Y$, and $\det D^2 u = f$ with the target density $1$ (`Transport(target)`).
It shares the network, the input convex `ICNN`, and changes the loss, the sampler, the start and the optimizer.

### The method

The loss (Section 3.3, eq. (3.4)) over the interior points $x_j$, the boundary points $y_k$ of $X$ and points $t_l$ on the boundary of $Y$ is

$$
\begin{aligned}
E = {} & (1/M) \sum_j (\det D^2 u(x_j) - f(x_j))^2 \cr
& + C \left[ (1/K) \sum_k \min_l \lvert \nabla u(y_k) - t_l \rvert^2 + (1/L) \sum_l \min_k \lvert \nabla u(y_k) - t_l \rvert^2 \right]
\end{aligned}
$$

with $C = 1$.
The bracket is a discrete form of the condition $\nabla u(\partial X) = \partial Y$, through the Hausdorff distance between the image of the boundary and the target's boundary: the first sum keeps each image near $\partial Y$, the second leaves no point of $\partial Y$ far from the images, so the map covers the target's boundary.
It needs no geometry of the target beyond points on its boundary, so any target that samples its boundary works; the paper's equation has $f / g(\nabla u)$ on the right, and with the uniform target density $1$ of a `Transport` problem that is $f$.
This is `HausdorffLoss(boundary_weight=1.0)`, its two halves logged as `into` and `onto`, from the kernel `functional.nearest_squared_distances`; `TransportSampler` draws the three point sets, $800$ each.

The network has $4$ hidden layers of $10$ softplus units, `ICNN(width=10, depth=4)`, $433$ parameters in 2D.
Before the training it is fitted so that $\nabla u$ approximates the identity map (Section 4.1, after Singh, Bauer and Joshi 2022), since an input convex network trains badly from a random start: `IdentityStart(network=ICNN(...))` fits the gradient of the module to $x$ by L-BFGS at the collocation points, up to $1000$ iterations, and records the fit in `result.info["network"]["identity_fit"]`.
The training takes $200$ Adam steps and then L-BFGS steps up to $2000$ steps in all (Figure 5; Table 1: strong Wolfe line search, step $1$, full batch), `Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200)`; `TransportICNN(adam_steps=2000)` is the paper's Adam-only schedule.
The preset checks the mass balance in its `check`: the integral of $f$ over the domain must equal the target's area.

```text
TransportICNN() = NeuralSolver(
    network=IdentityStart(network=ICNN(width=10, depth=4), fit_steps=1000, fit_tol=1e-6),
    loss=HausdorffLoss(boundary_weight=1.0),
    sampler=TransportSampler(interior_points=800, boundary_points=800, target_points=800),
    optimizer=Switch(first=Adam(lr=1e-3), then=LBFGS(), after=200),
    max_steps=2000,
    tol=1e-7,
)
```

### Source paper

A. Caboussat, A. Peruso, "Convex physics informed neural networks for the Monge-Ampere optimal transport problem", arXiv 2501.10162v2 (2025-08-16), the version read here; also in Eng. Comput. 43(8), a version we have not read.
Section 3.3 defines the loss, Section 4.1 the network, the start and the training (Table 1), and Section 4.2 the disk-to-ellipse problem (T12) with Table 2 (Adam against L-BFGS over ten runs) and Figures 2 to 6.
Sections 4.3 to 4.5 (an ellipse onto a rotated ellipse, Gaussian densities on the square, T13 to T15) are figures only, not compared.

### Parameters

| Part | Parameter | Default | From |
|---|---|---|---|
| `ICNN` | `width`, `depth` | $10$, $4$ | Section 4.1 |
| `ICNN` | `init_std` | $1$ | not given; as `DirichletICNN` |
| `ICNN` | `alpha` | $0$ | not in the paper; as above |
| `IdentityStart` | `fit_steps`, `fit_tol` | $1000$, $10^{-6}$ | not given |
| `HausdorffLoss` | `boundary_weight` | $1$ | $C = 1$, Section 4.1 |
| `TransportSampler` | `interior_points`, `boundary_points` | $800$, $800$ | $N_c = N_b = 800$, Section 4.1 |
| `TransportSampler` | `target_points` | $800$ | not given; as many as the boundary points |
| `Switch`, `Adam` | `adam_steps`, `adam_lr` | $200$, $10^{-3}$ | the switch from Figure 5; the learning rate not given, torch's default |
| `LBFGS` | `lr`, `history_size` | $1$, $100$ | Table 1; torch's default memory |
| `NeuralSolver` | `max_steps` | $2000$ | Table 1's $100$ epochs of $20$ iterations, Figure 5's $2000$ epochs |
| `NeuralSolver` | `tol` | $10^{-7}$ | torch's `tolerance_grad` |

`TransportICNN` takes these flat (`TransportICNN(adam_lr=1e-2, seed=3)`), with `seed` and `accelerator` as every neural solver.

### Differences from the paper

- One training step is one Adam step or one L-BFGS iteration; the paper calls $20$ L-BFGS iterations one epoch, and torch ends such an epoch early when the loss changes by less than $10^{-9}$, which our step budget does not.
- The paper does not say how it fits the start, nor the Adam learning rate, nor how it initialises the network before the fit, nor how many target points it takes and how it draws them; the defaults above fill these in, and `benchmarks/icnn.md` (Findings 3 and 4) shows where they matter.
- The target points are uniform in arc length on the target's boundary and independent of the boundary points.
- Float64 throughout (ADR 0008).

### Known limits

- With independent random points on the two boundaries, even the exact map leaves the transport term at about $10^{-4}$ on T12 with $800$ points each; this is the loss of one map, not a bound over maps, and the trained networks end within $4$% of it, above or below, with a map error of about $2 \times 10^{-3}$ (root mean square). Points on the two boundaries that the exact map pairs make the term zero and the error about seven times smaller, but that pairing needs the solution (`benchmarks/icnn.md`, Finding 3). More points on both boundaries lower the exact map's level; more target points alone leave its `onto` half in place.
- The transport term compares every image with every target point, $K L$ distances per step, so its time and memory grow with the product of the two counts.
- `check` integrates $f$ with `check_mass_balance`, which knows a `Box`, a `Disk` and an `Ellipse`, so the preset supports a problem only when its domain and its target are among these: on a `CutQuarterDisk` source or target, `supports` is False and `solve` raises `UnsupportedProblemError`. A composed `NeuralSolver` with the same parts runs there, and its caller checks the mass (for a `CutQuarterDisk` source onto a known target, with `check_mass_balance(problem, mass=...)`).
- The L-BFGS phase dominates the time: $14$ minutes per seed on an A100 for the receipt, ten seeds side by side; the Adam-only schedule takes under $2$ minutes.
- The network is convex by construction, so $\nabla u$ is a monotone map; it maps the boundary onto the target's boundary only as well as the loss enforces it.

## Reproduction

At the levels of ADR 0002, as [`benchmarks/icnn.md`](../../benchmarks/icnn.md) states them:

| Problem | Source | Level | Ours over seeds $0$ to $9$, median $[5, 95]$% |
|---|---|---|---|
| T1 | Nystrom-Vestberg 2023, Figure 2 (colorbar up to $2.97 \times 10^{-5}$) | figure only | max error $5.14 \times 10^{-5}$ $[2.02 \times 10^{-5}, 1.93 \times 10^{-4}]$ on the shared grid |
| T6 | Nystrom-Vestberg 2023, Table 1, $\mathrm{stdev} = 0$: max $8.58 \times 10^{-6}$, average $1.17 \times 10^{-6}$ at $10^6$ random interior points | mismatch | max $1.10 \times 10^{-4}$ $[5.49 \times 10^{-5}, 2.36 \times 10^{-4}]$, average $1.49 \times 10^{-5}$ $[8.63 \times 10^{-6}, 3.71 \times 10^{-5}]$ at the same kind of points |
| T12 (`TransportICNN`) | Caboussat-Peruso 2025, Table 2, L-BFGS row: L2 error of $\nabla u$ $2.67 \times 10^{-3}$ (train), $2.68 \times 10^{-3}$ (test), mean over $10$ runs | statistical, for the errors only | $2.06 \times 10^{-3}$ $[1.45 \times 10^{-3}, 3.78 \times 10^{-3}]$ (train), $2.05 \times 10^{-3}$ $[1.44 \times 10^{-3}, 3.78 \times 10^{-3}]$ (test), root mean square; the bootstrap band of our ten-seed mean, $[1.83 \times 10^{-3}, 2.75 \times 10^{-3}]$, holds $2.67 \times 10^{-3}$ too, but only just |
| T12 (`TransportICNN(adam_steps=2000)`) | the same table, Adam row: $2.40 \times 10^{-2}$ (train), $2.50 \times 10^{-2}$ (test); the final losses of both rows | mismatch | $7.45 \times 10^{-3}$ $[3.71 \times 10^{-3}, 2.15 \times 10^{-2}]$ (train); the final L-BFGS loss $1.38 \times 10^{-4}$ $[1.16 \times 10^{-4}, 1.66 \times 10^{-4}]$ against $1.13 \times 10^{-5}$ |

On T6 our errors are about $13$ times the paper's, and on T1 the figure's top lies near the lower end of our band: the paper's runs trained further, and `benchmarks/icnn.md` records what we found about why (Findings 1 and 2).
With a mean in place of the printed sum in the equation term, T6 is still a mismatch, about $3.6$ times above Table 1, though the runs meet the paper's stopping test (Finding 2); the preset and the level above stay with the print until #13 decides.
On T12 the L-BFGS errors match the paper at the statistical level, but its final loss lies ten times below both the level the exact map leaves at our independent boundary points and every trained seed of ours, and its Adam runs trained less far than ours (`benchmarks/icnn.md`, Findings 3 and 4).
The agreement is in the error columns only: the paper's figure is a mean over ten runs with a standard deviation larger than the mean, set against our per-seed band and against the band of our mean, and Finding 3 suggests that it may join two regimes and be in part a coincidence.
No value is pinned in `tests/test_reproduction.py`: a figure-only level gives no number to pin, a mismatch is not pinned, and a statistical level rests on ten GPU runs of $14$ minutes each, which the tests cannot repeat.
