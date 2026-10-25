# Penalty MLP

The penalty MLP of Nystrom and Vestberg (2023) is the first neural solver for the Dirichlet problem of $\det D^2 u = f$.
A feed-forward network inside an ansatz that meets the boundary condition is trained to satisfy the equation at collocation points, with a penalty on non-convexity that steers it to the convex solution.
maesolvers ships it as the preset `PenaltyMLP`, a `NeuralSolver` composed of four parts (ADR 0004), trained on the shared loop of ADR 0008.
Reproduction: [`benchmarks/penalty_mlp.md`](../../benchmarks/penalty_mlp.md).

## The method

The ansatz (the paper's eq. (3.1), after Berg and Nystrom 2018) is

$$
u(x) = G(x) + D(x) N(x)
$$

where $N$ is the trained network, $G$ a network fitted to the boundary data $g$, and $D$ a network fitted to the distance to the boundary, which vanishes there.
On the boundary $u - g = (G - g) + D N$: the misfit of $G$ plus what $N$ leaks through the residue of $D$, so $u$ meets the boundary condition only as well as $G$ fits $g$ and $D$ vanishes there, and the loss has no boundary term.
$G$ has one hidden layer of $20$ tanh units and is fitted to $g$ at the boundary collocation points; $D$ has one hidden layer of $40$ tanh units and is fitted to the exact distance at $300$ interior points and to $0$ at the boundary points.
Both fits minimise the mean squared error by L-BFGS until the largest gradient component is at most `fit_tol = 1e-5`, or for at most `fit_steps = 10000` iterations, and $G$ and $D$ stay fixed while $N$ trains (`HardDirichlet`).
On T1 and T6, seeds $0$ to $9$, both fits meet the tolerance, $G$ in at most $557$ iterations and $D$ in at most $2752$; `result.info["network"]` records, per fit, the iterations, the final largest gradient component, whether it converged and the misfit.
$N$ has five hidden layers of $10$ sigmoid units and a linear output, $481$ parameters in 2D, all drawn from the standard normal distribution at the start (`MLP`).

The loss (eqs. (3.3) to (3.5)) over the interior collocation points $x_1, \dots, x_M$ is

$$
E = \sum_j (\det D^2 u(x_j) - f(x_j))^2 + C (1/M) \sum_j (\min(0, u_{xx}(x_j))^2 + \min(0, u_{yy}(x_j))^2)
$$

with $C = 10^4$ (`PenaltyLoss`); the equation term is a sum, not a mean, as the paper prints it, so its weight against the penalty grows with $M$.
The derivatives are those of the whole ansatz, by automatic differentiation.
The penalty works because the equation term makes $\det D^2 u$ close to $f \ge 0$: a symmetric $2 \times 2$ matrix with a nonnegative determinant is positive semidefinite exactly when both diagonal entries are nonnegative.
In three dimensions it fails: the matrix with $1$ on the diagonal and $2$ everywhere else has determinant $5$ and eigenvalues $5$, $-1$ and $-1$.
So the method is 2D only, as the paper says (Section 3.1).

The collocation points are uniform random: $3000$ inside the domain by rejection from the bounding box, $400$ on the boundary with each side getting points in proportion to its length (`UniformSampler`).
They are drawn from the seed by the sampler; whether ADR 0008's shared collocation set is per problem or per sampler, and whether it is stored, is open (#52).
The optimizer is L-BFGS with a strong Wolfe line search, one quasi-Newton iteration per training step (`LBFGS`).
The training stops when the largest component of the loss gradient is at most `tol = 1e-5`, the default of SciPy's BFGS, or after `max_steps = 5000` steps.
It runs on the CPU unless `accelerator` says otherwise.

## Source paper

K. Nystrom, M. Vestberg, "Solving the Dirichlet problem for the Monge-Ampere equation using neural networks", J. Comput. Math. Data Sci. 8 (2023) 100080; arXiv 2110.03310 (v3, 2023-06-13, the version read here).
Section 3.1 defines the method and Example 4.1 runs it on T1, with Figure 1 as its only result; every later example and every table of the paper uses the paper's other method, an input convex network (Section 3.2).
`benchmarks/penalty_mlp.md` records this finding and the comparison.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `PenaltyMLP` | `width`, `depth` | $10$, $5$ | units per hidden layer and hidden layers of $N$ |
| `PenaltyMLP` | `convexity_weight` | $10^{4}$ | $C$, the weight of the penalty |
| `PenaltyMLP` | `interior_points`, `boundary_points` | $3000$, $400$ | collocation points inside and on the boundary (`UniformSampler`) |
| `PenaltyMLP` | `fit_steps`, `fit_tol` | $10000$, $10^{-5}$ | most L-BFGS iterations of each fit of $G$ and $D$, and the largest gradient component at which a fit stops (`HardDirichlet`) |
| `NeuralSolver` | `max_steps` | $5000$ (preset), $1000$ | most training steps |
| `NeuralSolver` | `tol` | $10^{-5}$ | largest gradient component at which the training stops, converged |
| `NeuralSolver` | `seed` | $0$ | seeds the collocation points and the initial networks |
| `NeuralSolver` | `accelerator` | `"cpu"` | `"cpu"`, `"gpu"`, or `"auto"` (a GPU when there is one) |
| `HardDirichlet` | `lift_width`, `distance_width` | $20$, $40$ | hidden units of $G$ and $D$ |
| `HardDirichlet` | `distance_points` | $300$ | interior points at which $D$ is fitted |
| `PenaltyLoss` | `boundary_weight` | $0$ | weight of a mean squared boundary misfit, for a network without the ansatz |
| `LBFGS` | `lr`, `history_size` | $1$, $100$ | initial step of the line search, curvature pairs kept |

The preset takes the parameters of its parts flat: `PenaltyMLP(width=20, seed=3)` is `NeuralSolver(network=HardDirichlet(network=MLP(width=20)), loss=PenaltyLoss(), sampler=UniformSampler(), optimizer=LBFGS(), max_steps=5000, seed=3)`.
`docs/guide.md` shows how to swap a part or write one.

## Differences from the paper

- L-BFGS with $100$ curvature pairs replaces SciPy's full BFGS; the paper's restarts with Adam at a learning rate of about $10^{-7}$ when BFGS stopped on a line-search failure are not done: a step that changes nothing ends the training (`stop = "stalled"`).
- The paper does not give the number of BFGS iterations for Example 4.1; the preset stops at $5000$, the budget of its Example 4.6.
- The paper does not give the standard deviation of the initial weights; the preset uses $1$, TensorFlow's default for a normal draw.
- The fits of $G$ and $D$ stop when the largest gradient component is at most $10^{-5}$, SciPy's default test (`norm=inf`); the paper's text says the Euclidean norm, which is stricter by up to the square root of the number of parameters. We follow SciPy, which the paper ran.
- The paper caps neither fit (SciPy's own default cap is $200$ iterations per parameter, $16200$ for $G$ and $32200$ for $D$); the preset caps each at $10000$, which neither fit reaches on T1 or T6.
- $G$ and $D$ are fitted by L-BFGS with $100$ curvature pairs, not by SciPy's full BFGS.
- $D$ is fitted to the exact distance on every supported domain; the paper does so on the unit square, where the distance is known, and refers to Berg-Nystrom 2018 for other domains.

## Known limits

- 2D only, Dirichlet only, and only on domains that sample their boundary and know the distance to it (`Box`, `Disk`, `CutQuarterDisk`): `solve` raises `UnsupportedProblemError` for T9, T10, T12 and an `Ellipse`.
- The max error lies on the boundary, where $u - g = (G - g) + D N$. The fits stop on the gradient of a mean square, not on the misfit, so even converged $G$ misses $g$ and $D$ misses $0$ by about $10^{-2}$; the leak $D N$ then grows with $\lvert N \rvert$, and in the seeds that train worst it is the larger term. With `fit_tol = 1e-7` the boundary error on T1 falls about fourfold and the max error moves into the interior in most seeds (`benchmarks/penalty_mlp.md`, Finding 2); the preset keeps the paper's $10^{-5}$.
- Convexity is penalised at the collocation points only; `evaluate` reports the non-convex fraction of the trained network on the shared grid.
- The penalty excludes a saddle only where the equation term makes $\det D^2 u$ positive, that is where $f > 0$. Where $f = 0$ on a set (T4, T5 and T20), a Hessian with a slightly negative determinant and a nonnegative diagonal is a saddle that the loss does not penalise; there `nonconvex_fraction` from `evaluate` is the only check.
- The training is deterministic only on one machine with one torch thread count: CPU and GPU, two GPU models, or two thread counts round differently, and the runs drift apart (the fits of $G$ and $D$ already differ between $1$ and $4$ threads). `result.info` records `device` and `threads`; the receipts use one thread per seed.
- The networks are small, so a GPU is launch-bound: on a100server a training step takes about $70$ ms on an A100 used by one process, about $200$ ms when four processes share it, and about $70$ ms on one CPU thread (`benchmarks/penalty_mlp.md`, Device).

## Reproduction

At the levels of ADR 0002, as [`benchmarks/penalty_mlp.md`](../../benchmarks/penalty_mlp.md) states them:

| Problem | Source | Level | Ours, max error over seeds $0$ to $9$, median $[5, 95]$% |
|---|---|---|---|
| T1 | Nystrom-Vestberg 2023, Figure 1 (colorbar up to about $5.7 \times 10^{-3}$) | figure only | $7.06 \times 10^{-3}$ $[4.57 \times 10^{-3}, 1.66 \times 10^{-2}]$ on the shared grid |
| T6 | Nystrom-Vestberg 2023, Table 1, which is the ICNN, not this method | not a reproduction | $5.45 \times 10^{-3}$ $[3.46 \times 10^{-3}, 9.42 \times 10^{-3}]$ on the shared grid, against $8.58 \times 10^{-6}$ for the ICNN |

No value is pinned in `tests/test_reproduction.py`: a figure-only level gives no number to pin.
