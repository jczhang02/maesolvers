# Input convex network (ICNN)

The input convex network of Nystrom and Vestberg (2023) solves the Dirichlet problem of `det D^2 u = f` with a network that is convex in its input by construction, so the solution it finds is convex whatever the training does.
The boundary condition enters the loss as a penalty.
maesolvers ships it as the network part `ICNN` and the preset `DirichletICNN`, a `NeuralSolver` composed of four parts (ADR 0004), trained on the shared loop of ADR 0008.
Reproduction: [`benchmarks/icnn.md`](../../benchmarks/icnn.md).

## The method

The network (the paper's eq. (3.7), after Amos, Xu and Kolter 2017) with `k` hidden layers is

```text
z_1     = s(L_0 x + b_0)
z_{j+1} = s(W_j z_j + L_j x + b_j),   j = 1, ..., k - 1
u(x)    = W_k z_k + L_k x + b_k
```

where `s(t) = log(1 + e^t)` is the softplus, the `L_j` and `b_j` are free, and every `W_j` has non-negative entries.
Each `z_j` is convex in `x`: an affine map of `x` is convex, a non-negative combination of convex functions is convex, and a convex non-decreasing function of a convex function is convex.
So `u` is convex for every value of the parameters, in any dimension, and its Hessian is positive semidefinite at every point, not only at the collocation points.
The softplus is smooth, so `u` has a Hessian everywhere; ReLU is convex but has no second derivative, and tanh and the sigmoid are not convex, so `ICNN` offers no other activation.
As in the paper (Section 4), `W_j` is the elementwise square `V_j^2` of the trained parameter `V_j`, and every parameter starts from the normal distribution, with standard deviation `init_std = 1`.
With five hidden layers of 10 units in 2D the network has 563 parameters (Example 4.2): 30 in the first layer, 130 in each of the four later hidden layers, 13 in the output layer.

A convex network cannot use the ansatz `u = G + D N` of the penalty MLP (`docs/methods/penalty_mlp.md`): the product `D N` of convex functions need not be convex.
So the boundary condition is a penalty, and the loss (eq. (3.8), as Example 4.2 uses it) over the interior collocation points `x_1, ..., x_M` and the boundary points `y_1, ..., y_K` is

```text
E = sum_j (det D^2 u(x_j) - f(x_j))^2 + sum_k (det D^2 u(y_k) - f(y_k))^2 + C (1/K) sum_k (u(y_k) - g(y_k))^2
```

with `C = 100`: the equation term is a sum, not a mean, and in Example 4.2 it runs over the boundary points too.
The preset follows the print; a mean would be the sum with `C = 3.4e5`, up to a factor `1/3400`, and our diagnostic runs side with the mean (Known limits, below).
This is `PenaltyLoss(convexity_weight=0, boundary_weight=100, equation_at_boundary=True)`.
So the ICNN differs from the penalty MLP in its loss as well as in its network: no convexity penalty, a boundary penalty instead of the ansatz, and the equation at the boundary points.
It differs in its optimizer too, below.

The collocation points are uniform random, 3000 inside the domain and 400 on the boundary, each piece of the boundary getting points in proportion to its length (`UniformSampler`), drawn from the seed by the sampler; whether ADR 0008's shared collocation set is per problem or per sampler, and whether it is stored, is open (#52).
The optimizer is BFGS with a dense inverse Hessian approximation, one iteration of SciPy's `minimize(method="BFGS")`, which the paper ran, per training step (`BFGS`): the approximation starts as the identity and gets the BFGS update after every step, and the step comes from SciPy's own strong Wolfe line search, so the iterates are SciPy's up to rounding.
The limited-memory `LBFGS` of the penalty MLP trains this network far less well: on T1 it leaves the loss about 50000 times higher after 10000 steps, and the max error about 270 times higher (`benchmarks/icnn.md`, Finding 1).
The training stops when the largest component of the loss gradient is at most `tol = 1e-5`, the default of SciPy's BFGS, or after `max_steps = 10000` steps.
It runs on the CPU unless `accelerator` says otherwise.

## Source paper

K. Nystrom, M. Vestberg, "Solving the Dirichlet problem for the Monge-Ampere equation using neural networks", J. Comput. Math. Data Sci. 8 (2023) 100080; arXiv 2110.03310 (v3, 2023-06-13, the version read here).
Section 3.2 defines the method and Example 4.2 runs it on T1, with Figure 2 as its result; Example 4.6 runs it on T6, with Table 1.
The network construction is that of B. Amos, L. Xu, J. Z. Kolter, "Input convex neural networks", ICML 2017.
The other ICNN of the roadmap, Caboussat and Peruso (2025), trains an ICNN on the transport boundary condition; it comes with transport support in 0.2 (#29).

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `DirichletICNN` | `width`, `depth` | 10, 5 | units per hidden layer and hidden layers of the network (`ICNN`) |
| `DirichletICNN` | `init_std` | 1 | standard deviation of every initial parameter of the network (`ICNN`) |
| `DirichletICNN` | `boundary_weight` | 100 | `C`, the weight of the boundary misfit (`PenaltyLoss`), positive: without the ansatz, this term alone imposes `g` |
| `DirichletICNN` | `interior_points`, `boundary_points` | 3000, 400 | collocation points inside and on the boundary (`UniformSampler`) |
| `NeuralSolver` | `max_steps` | 10000 (preset), 1000 | most training steps |
| `NeuralSolver` | `tol` | 1e-5 | largest gradient component at which the training stops, converged |
| `NeuralSolver` | `seed` | 0 | seeds the collocation points and the initial network |
| `NeuralSolver` | `accelerator` | `"cpu"` | `"cpu"`, `"gpu"`, or `"auto"` (a GPU when there is one) |
| `PenaltyLoss` | `convexity_weight`, `equation_at_boundary` | 0, True (preset) | the penalty on negative `u_xx`, `u_yy`, which a convex network does not need, and whether the equation is also summed at the boundary points |
| `BFGS` | none | | |

The preset takes the parameters of its parts flat: `DirichletICNN(width=20, seed=3)` is `NeuralSolver(network=ICNN(width=20), loss=PenaltyLoss(convexity_weight=0.0, boundary_weight=100.0, equation_at_boundary=True), sampler=UniformSampler(), optimizer=BFGS(), max_steps=10_000, seed=3)`.
`ICNN` also composes with the other parts; `equation_at_boundary=False` gives the loss of the paper's Section 5, whose equation term runs over the interior points only.
`docs/guide.md` shows how to swap a part or write one.

## Differences from the paper

- The paper restarted with Adam when BFGS stopped on a failed line search (Example 4.1); the preset does not: a step that changes nothing ends the training (`stop = "stalled"`).
- The paper gives the iterations of one T1 run (convergence after 2287) and caps the T6 runs of Table 1 at 10000; the preset caps every run at 10000 steps.
- The paper does not give the standard deviation of the initial weights; the preset uses 1 for every parameter, the roots `V_j` included, as the paper's "normal distribution" reads literally (TensorFlow 1's default for a normal draw; TensorFlow 2's `RandomNormal` defaults to 0.05). Under the dense BFGS the choice matters less than the optimizer (`benchmarks/icnn.md`, Finding 1).
- The training stops when the largest gradient component is at most 1e-5, SciPy's default test (`norm=inf`); the paper's text says the Euclidean norm.

## Known limits

- Dirichlet only, and only on domains that sample their boundary (`Box` in any dimension, `Disk`, `CutQuarterDisk`): `solve` raises `UnsupportedProblemError` for T12, T16 and an `Ellipse`. The transport ICNN of Caboussat-Peruso 2025 is #29.
- The boundary condition is a penalty, so `u` meets it only approximately: on T1 and T6 the max error lies on the boundary in every seed.
- Within the 10000-step budget no seed meets the stopping test, and on T6 the errors are about 13 times those of the paper's Table 1 (`benchmarks/icnn.md`, T6).
  The printed sum is the likely cause: read as a mean, the equation term weighs the boundary misfit, where the max error is, 3400 times more, and `DirichletICNN(boundary_weight=340_000, tol=0.034)`, the mean reading up to the BFGS start, puts Table 1 inside its band on T6 (max 5.47e-6 [2.58e-6, 1.68e-5] against 8.58e-6) and meets the stopping test in 8 of 10 seeds after about 3000 steps (`benchmarks/icnn.md`, Finding 2).
  The preset keeps the sum until #13 decides.
- Convexity holds by construction, but the class of functions is narrower than that of a plain network of the same size: every hidden unit is convex, so a solution that is nearly affine in some direction, or whose Hessian degenerates, as for `f = 0` on a set (T4, T5, T20), is harder to represent.
- With the standard normal start, the squared roots have mean 1, so the activations grow from layer to layer and the first loss is very large (about 6e19 on T1 for seed 0); BFGS brings it down within a few hundred steps, but the start is far from the solution.
  The first loss grows with the dimension, since every layer reads all `d` inputs: at the defaults and seeds 0 to 2 it is 6e19 to 1e21 on T1 and T6 (2D), 2e26 to 3e28 on T9 (3D) and 2e33 to 3e36 on T10 (4D).
- `DirichletICNN` runs on T9 and T10 (3D and 4D Dirichlet problems), but its accuracy there is not measured: the paper's Tables 2 and 3 are not reproduced (`benchmarks/icnn.md`, Not compared), and the tests train only a few steps in 3D.
- `BFGS` stores a dense `n x n` matrix and multiplies by it at every step, so it suits networks of a few thousand parameters at most; for a larger `ICNN`, pass `LBFGS()` and expect slower training.
- The training is deterministic only on one machine with one torch thread count; `result.info` records `device` and `threads`, and the receipts use one thread per seed.

## Reproduction

At the levels of ADR 0002, as [`benchmarks/icnn.md`](../../benchmarks/icnn.md) states them:

| Problem | Source | Level | Ours over seeds 0 to 9, median [5%, 95%] |
|---|---|---|---|
| T1 | Nystrom-Vestberg 2023, Figure 2 (colorbar up to 2.97e-5) | figure only | max error 5.14e-5 [2.02e-5, 1.93e-4] on the shared grid |
| T6 | Nystrom-Vestberg 2023, Table 1, `stdev = 0`: max 8.58e-6, average 1.17e-6 at 10^6 random interior points | mismatch | max 1.10e-4 [5.49e-5, 2.36e-4], average 1.49e-5 [8.63e-6, 3.71e-5] at the same kind of points |
| T12 | Caboussat-Peruso 2025, Table 2 | not compared: needs transport boundary conditions (0.2, #29) | |

On T6 our errors are about 13 times the paper's, and on T1 the figure's top lies near the lower end of our band: the paper's runs trained further, and `benchmarks/icnn.md` records what we found about why (Findings 1 and 2).
With a mean in place of the printed sum in the equation term, T6 reaches the statistical level (Finding 2); the preset and the level above stay with the print until #13 decides.
No value is pinned in `tests/test_reproduction.py`: a figure-only level gives no number to pin, and a mismatch is not pinned.
