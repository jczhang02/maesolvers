# Penalty MLP

The penalty MLP of Nystrom and Vestberg (2023) is the first neural solver for the Dirichlet problem of `det D^2 u = f`.
A feed-forward network inside an ansatz that meets the boundary condition is trained to satisfy the equation at collocation points, with a penalty on non-convexity that steers it to the convex solution.
maesolvers ships it as the preset `PenaltyMLP`, a `NeuralSolver` composed of four parts (ADR 0004), trained on the shared loop of ADR 0008.
Reproduction: [`benchmarks/penalty_mlp.md`](../../benchmarks/penalty_mlp.md).

## The method

The ansatz (the paper's eq. (3.1), after Berg and Nystrom 2018) is

```text
u(x) = G(x) + D(x) N(x)
```

where `N` is the trained network, `G` a network fitted to the boundary data `g`, and `D` a network fitted to the distance to the boundary, which vanishes there.
So `u` meets the boundary condition as well as `G` fits `g`, whatever `N` is, and the loss needs no boundary term.
`G` has one hidden layer of 20 tanh units and is fitted to `g` at the boundary collocation points; `D` has one hidden layer of 40 tanh units and is fitted to the exact distance at 300 interior points and to 0 at the boundary points.
Both fits minimise the mean squared error by L-BFGS until the largest gradient component is at most 1e-5, and `G` and `D` stay fixed while `N` trains (`HardDirichlet`).
`N` has five hidden layers of 10 sigmoid units and a linear output, 481 parameters in 2D, all drawn from the standard normal distribution at the start (`MLP`).

The loss (eqs. (3.3) to (3.5)) over the interior collocation points `x_1, ..., x_M` is

```text
E = sum_j (det D^2 u(x_j) - f(x_j))^2 + C (1/M) sum_j (min(0, u_xx(x_j))^2 + min(0, u_yy(x_j))^2)
```

with `C = 1e4` (`PenaltyLoss`); the equation term is a sum, not a mean, as the paper prints it, so its weight against the penalty grows with `M`.
The derivatives are those of the whole ansatz, by automatic differentiation.
The penalty works because the equation term makes `det D^2 u` close to `f >= 0`: a symmetric 2 x 2 matrix with a nonnegative determinant is positive semidefinite exactly when both diagonal entries are nonnegative.
In three dimensions it fails: the matrix with 1 on the diagonal and 2 everywhere else has determinant 5 and eigenvalues 5, -1 and -1.
So the method is 2D only, as the paper says (Section 3.1).

The collocation points are uniform random: 3000 inside the domain by rejection from the bounding box, 400 on the boundary with each side getting points in proportion to its length (`UniformSampler`).
The optimizer is L-BFGS with a strong Wolfe line search, one quasi-Newton iteration per training step (`LBFGS`).
The training stops when the largest component of the loss gradient is at most `tol = 1e-5`, the default of SciPy's BFGS, or after `max_steps = 5000` steps.

## Source paper

K. Nystrom, M. Vestberg, "Solving the Dirichlet problem for the Monge-Ampere equation using neural networks", J. Comput. Math. Data Sci. 8 (2023) 100080; arXiv 2110.03310 (v3, 2023-06-13, the version read here).
Section 3.1 defines the method and Example 4.1 runs it on T1, with Figure 1 as its only result; every later example and every table of the paper uses the paper's other method, an input convex network (Section 3.2).
`benchmarks/penalty_mlp.md` records this finding and the comparison.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `PenaltyMLP` | `width`, `depth` | 10, 5 | units per hidden layer and hidden layers of `N` |
| `PenaltyMLP` | `convexity_weight` | 1e4 | `C`, the weight of the penalty |
| `PenaltyMLP` | `interior`, `boundary` | 3000, 400 | collocation points inside and on the boundary |
| `NeuralSolver` | `max_steps` | 5000 (preset), 1000 | most training steps |
| `NeuralSolver` | `tol` | 1e-5 | largest gradient component at which the training stops, converged |
| `NeuralSolver` | `seed` | 0 | seeds the collocation points and the initial networks |
| `NeuralSolver` | `accelerator` | `"auto"` | `"cpu"`, `"gpu"`, or a GPU when there is one |
| `HardDirichlet` | `lift_width`, `distance_width` | 20, 40 | hidden units of `G` and `D` |
| `HardDirichlet` | `distance_points` | 300 | interior points at which `D` is fitted |
| `HardDirichlet` | `fit_steps` | 1000 | most L-BFGS iterations of each fit |
| `PenaltyLoss` | `boundary_weight` | 0 | weight of a mean squared boundary misfit, for a network without the ansatz |
| `LBFGS` | `lr`, `history_size` | 1, 100 | initial step of the line search, curvature pairs kept |

The preset takes the parameters of its parts flat: `PenaltyMLP(width=20, seed=3)` is `NeuralSolver(network=HardDirichlet(network=MLP(width=20)), loss=PenaltyLoss(), sampler=UniformSampler(), optimizer=LBFGS(), max_steps=5000, seed=3)`.
`docs/guide.md` shows how to swap a part or write one.

## Differences from the paper

- L-BFGS with 100 curvature pairs replaces SciPy's full BFGS; the paper's restarts with Adam at a learning rate of about 1e-7 when BFGS stopped on a line-search failure are not done: a step that changes nothing ends the training (`stop = "stalled"`).
- The paper does not give the number of BFGS iterations for Example 4.1; the preset stops at 5000, the budget of its Example 4.6.
- The paper does not give the standard deviation of the initial weights; the preset uses 1, TensorFlow's default for a normal draw.
- `D` is fitted to the exact distance on every supported domain; the paper does so on the unit square, where the distance is known, and refers to Berg-Nystrom 2018 for other domains.

## Known limits

- 2D only, Dirichlet only, and only on domains that sample their boundary and know the distance to it (`Box`, `Disk`, `CutQuarterDisk`): `solve` raises `UnsupportedProblemError` for T9, T10, T12 and an `Ellipse`.
- The boundary error is the error of the fit of `G` to `g`, a network of 20 units, and it does not shrink as `N` trains.
- Convexity is penalised at the collocation points only; `evaluate` reports the non-convex fraction of the trained network on the shared grid.
- The training is not deterministic across devices: CPU and GPU, or two GPU models, round differently and the runs drift apart.
- The networks are small, so a GPU is launch-bound: on a100server a training step takes about 70 ms on an A100 used by one process, about 200 ms when four processes share it, and about 70 ms on one CPU thread (`benchmarks/penalty_mlp.md`, Device).

## Reproduction

At the levels of ADR 0002, as [`benchmarks/penalty_mlp.md`](../../benchmarks/penalty_mlp.md) states them:

| Problem | Source | Level | Ours, max error over seeds 0 to 9, median [5%, 95%] |
|---|---|---|---|
| T1 | Nystrom-Vestberg 2023, Figure 1 (colorbar up to about 5.7e-3) | figure only | 7.59e-3 [4.69e-3, 2.22e-2] on the shared grid |
| T6 | Nystrom-Vestberg 2023, Table 1, which is the ICNN, not this method | not a reproduction | 5.31e-3 [3.18e-3, 9.25e-3] on the shared grid, against 8.58e-6 for the ICNN |

No value is pinned in `tests/test_reproduction.py`: a figure-only level gives no number to pin.
