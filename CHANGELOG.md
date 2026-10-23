# Changelog

Every change to the public interface is listed here; 0.x may break the interface, but never silently.

## Unreleased

### Added

- `maesolvers.core.transport`, the shared helpers of the second boundary value problem (issue #26), pure numpy functions exported from `maesolvers.core`, for the target sets `Box` (any dimension), `Disk` and `Ellipse` (`TypeError` for another set; the `Domain` protocol is unchanged):
  `signed_distance(target, y)`, the signed distance to the target's boundary, positive outside and negative inside (`H` of Benamou-Froese-Oberman 2014, eq. (3));
  `support_function(target, n)`, `sup_{y in Y} y . n` for any direction `n` (their `H*`, eq. (5));
  `closest_boundary_point(target, y)`, the nearest boundary point and the outward unit normal there (Eberly's bisection for the ellipse, accurate to rounding near its centre and its major axis);
  `volume(domain)`; `integrate(domain, f, points=64)`, Gauss quadrature over a box, a disk or an ellipse;
  `check_mass_balance(problem, rtol=1e-3, points=64, *, mass=None)`, which raises `ValueError` unless the integral of `f` over the domain (computed by `integrate`, or `mass` when given) equals the target's volume, for a transport solver to call from its own `check`.
- `gradient_fd(u, points, step=None)` and `gradient_map_error(gradient, exact)` in `maesolvers.metrics`: the central-difference gradient of a field, and the max and RMS of `|grad u - grad u_exact|` as `gradient_max` and `gradient_rms`, for the tables that report the error of the transport map; `evaluate` and its fields are unchanged (ADR 0007).
- `gradient(u, x, create_graph=True)`, the autograd gradient, and `signed_distance(target, y)`, the torch form of the core function with the outward normal as its gradient in `y`, in `maesolvers.neural.functional`.
- `Batch.target`, the target set of a `Transport` problem for a loss on the transport condition, None for another boundary condition; a keyword field with default None after `f_boundary`, so existing code that builds a `Batch` keeps working, and `Batch.to` leaves it as it is.

## 0.1.0 (2026-10-23)

This release freezes `Problem`, `Result` and `evaluate` (ADR 0007): the research code builds on them, so from 0.1 on they change only together with a change to ADR 0007, listed here.
The rest of the interface (solvers, schemes, iterations, the neural parts and the presets) may still change in 0.x, never silently.

### Added

- `Monotone17(n, stencil=17, boundary="narrow", jacobian_floor=1e-12, spline_degree=3)`, the monotone wide-stencil scheme of Froese-Oberman 2011 on a 2D box with equal sides, with the 9-, 17- and 33-point stencils (`stencil`, typed `Literal[9, 17, 33]`, so the CLI lists the choices), exported at the top level and from `maesolvers.classical`.
  `discretize` raises `ValueError` for an `f` that is negative or not finite at an interior node.
  `spline_degree` is 1 to 5 whatever `n`; with fewer than `spline_degree` cells per side the spline takes degree `n`, as for `Standard9`.
  It provides `residual`, `jacobian` (a generalised Jacobian, so `Newton` on it is a semi-smooth Newton method) and `solve_laplace`, so `Newton(scheme=Monotone17(...))` solves it; `PoissonIteration` refuses it, as it has no `second_derivatives`.
  `directional_differences(grid, v)` returns the second differences of every pair at the interior nodes, shape `(pairs, 2, m)` in the order of `stencil_pairs`, for code that inspects a solution pair by pair.
  Below zero each pair's product continues with the slope `kappa = sqrt(max(f, mean f, 1))`, so a root has no negative second difference in a pair that enters the minimum, and Newton's stopping rule cannot accept a non-convex iterate that is not a root; `boundary` chooses between leaving the wide pairs out next to the boundary (`"narrow"`, which matches the published tables) and shortening their arms to the boundary (`"interpolate"`, which the papers describe).
  The pure kernels `stencil_pairs`, `directional_difference`, `pair_values` and `pair_jacobian_coefficients` are in `maesolvers.classical.monotone`, with `MonotoneGrid`, the context `Monotone17.discretize` returns, whose fields may change with the scheme.
- The preset `FDMonotone(n, stencil, boundary, jacobian_floor, spline_degree, max_iter=200, tol=1e-10)`, `Newton` on `Monotone17`, registered in `SOLVERS`; for a box with unequal sides its `UnsupportedProblemError` names the equal sides.
- `Hybrid(n, stencil=17, boundary="narrow", jacobian_floor=1e-12, spline_degree=3, epsilon=None, band=1.0)`, the hybrid scheme of Froese-Oberman 2011 (J. Comput. Phys.) on a 2D box with equal sides, exported at the top level and from `maesolvers.classical`: the weighted average `w MA_N + (1 - w) MA_M` of `Standard9` and `Monotone17` on one grid, in its residual and its Jacobian, so `Newton(scheme=Hybrid(...))` solves it.
  The schemes are its read-only attributes `standard` and `monotone`, built from the flat fields.
  `epsilon` must lie in `(0, 1)`: with `epsilon >= 1` every node would be singular.
  `Hybrid.weights(problem, nodes, h)` returns `w` at the unknowns, given the interior nodes in the order of the unknowns (shape `((n - 1)^2, 2)`) and the grid step; by default it is the paper's switching rule: 0 within `band` grid steps of the boundary and of the nodes where `f <= epsilon` or `f >= 1/epsilon` (`epsilon` None means `h`), 1 elsewhere; a subclass that overrides it plugs in a rule of its own, and `discretize` raises `ValueError` if it returns the wrong shape or a value outside `[0, 1]`.
  Its results carry `info["monotone_fraction"]`, the mean of `1 - w` over the unknowns (the share of unknowns with `w = 0` for a rule of 0 and 1).
  The pure kernel `hybrid_weights(f, epsilon, band)` is in `maesolvers.classical.hybrid`, with `HybridGrid`, the context `Hybrid.discretize` returns, whose fields may change with the scheme.
- `Newton(..., start=None)`: `start` is any `SchemeIteration` whose scheme has the same unknowns, such as `FDMonotone(n)` or `PoissonIteration(scheme=Standard9(n), max_iter=1)` (the start of Froese-Oberman 2011, without its convex envelope).
  `Newton.run` first runs it with its own `max_iter` and `tol` (its `setup` and `run`, not its `finalize`) and steps from its final unknowns instead of calling `initial_guess`; it raises `ValueError` if the unknowns differ in number, and warns (`RuntimeWarning`) if the start did not converge.
  `Result.info["iterations"]`, `converged`, `history` and the callbacks cover the steps on `scheme` only; `info["start_iterations"]` and `info["start_converged"]` report the start, and `time` includes it.
  `supports` and `check` ask `start` too.
- The preset `FDHybrid(n, stencil, boundary, jacobian_floor, spline_degree, epsilon, band, max_iter=200, tol=1e-10)`, `Newton` on `Hybrid` with `start=FDMonotone(...)` built from the same fields and `tol` (not a parameter; the start keeps its budget of 200 steps whatever `max_iter`), registered in `SOLVERS`; for a box with unequal sides its `UnsupportedProblemError` names the equal sides, as `FDMonotone`'s does.
- `Trace.info`, a mapping (empty by default) of entries for `Result.info` that only `run` knows; `Solver.solve` merges it with `Solution.info` and raises `ValueError` if it sets a key of the base class or a key `Solution.info` sets too.
- `IterativeSolver.run_from(problem, context, callbacks, iterate)`, the loop of `IterativeSolver.run` from a given first iterate, for a subclass whose first iterate needs more than the context (`Newton` with a `start`).
- `Filtered(accurate, monotone, epsilon=None)`, the filtered scheme of Froese-Oberman 2013: `F_M + eps S((F_A - F_M) / eps)` with any accurate and any monotone scheme that share their unknowns (`discretize` raises `ValueError` if their right-hand sides differ by more than a relative `1e-12`, construction if both have an `n` and they differ), exported at the top level and from `maesolvers.classical`.
  It provides `residual`, `jacobian` (the paper's approximation `(1 - S') J_M + max(S', 0) J_A`) and `solve_laplace` (the monotone part's), so `Newton` solves it; it supports a problem when both parts do.
  `epsilon=None` takes the paper's width `sqrt(h) + dtheta / 10`, with `h` the bounding box's longest side over `monotone.n` and `dtheta = 2 * monotone.angular_resolution`, the largest angle between neighbouring stencil directions, which the paper's T2 columns fit (`benchmarks/filtered.md`, Finding 3); a monotone part without `n` or `angular_resolution` needs an explicit `epsilon` (`TypeError` otherwise).
  Results carry the accurate part's solution fields and add `info["filter_width"]` and `info["monotone_fraction"]`, the share of unknowns where `|F_A - F_M| > eps`; `info["residual_max"]` is the filtered residual's.
  The pure kernels `filter_function`, `filter_slope` and `filter_width` are in `maesolvers.classical.filtered`, with `FilteredGrid`, the context `Filtered.discretize` returns, whose fields may change with the scheme.
- The preset `FDFiltered(n, stencil, boundary, jacobian_floor, epsilon=None, spline_degree, max_iter=200, tol=1e-10)`, `Newton` on `Filtered(accurate=Standard9(n), monotone=Monotone17(n, ...))`, registered in `SOLVERS`; its support and its equal-sides message are `FDMonotone`'s.
- `angular_resolution(points)` in `maesolvers.classical.monotone`, the angular resolution `dtheta` of the 9-, 17- and 33-point stencils (Oberman 2008, eq. (3.3): half the largest gap between directions; `Filtered` takes twice it), and the property `Monotone17.angular_resolution`.
- `Standard9` discretises 2D Dirichlet problems on a `Disk`, an `Ellipse` and the `CutQuarterDisk` of T6, with Shortley-Weller stencils next to the boundary, so `FDNewton` and `FDPoisson` solve T8 and T6 (issue #1); its `domains` (and the presets') are `(Box, Disk, Ellipse, CutQuarterDisk)`.
  On these domains `n` counts cells per side of the grid over the bounding box, `Result.nodes` lists the nodes with unknowns and then the boundary points, and the solution off the nodes is the Clough-Tocher interpolant (`spline_degree` applies to a box only), extrapolated to first order past the hull of the nodes so that it stays second order at corners off the grid.
- `shortley_weller_operators(domain, n)` in `maesolvers.classical.schemes`: the pure function behind it, returning a `ShortleyWellerOperators` named tuple (`nodes`, `interior`, `dxx`, `dyy`, `dxy`, `min_step`): the nodes, the mask of the nodes with unknowns, the operators `D_xx`, `D_yy`, `D_xy` and the smallest step fraction at each node.
- `Standard9` results (and so `FDNewton` and `FDPoisson` results) carry `info["min_step"]`, the smallest step fraction of a stencil (1 on a box), and `info["interpolant"]`, `"spline(k)"` or `"clough_tocher"`, the interpolant behind `Result.u` and `max_grid`.
- `maesolvers.neural`, the neural solvers (ADR 0004, 0008), behind the `neural` extra (torch, lightning >= 2): the package and every part import with numpy alone, so `import maesolvers` still loads no extra, and a solve without torch or lightning raises `ImportError` naming the extra.
  `NeuralSolver(network, loss, sampler, optimizer, max_steps=1000, tol=1e-5, seed=0, accelerator="cpu")` trains on a Lightning `Trainer` in float64, full batch, one optimizer step per training step; it stops converged when the largest component of the loss gradient is at most `tol`, and otherwise after `max_steps`, on a step that changes nothing (not counted as a step), or on a non-finite loss.
  A trainable parameter the loss does not reach (the output bias of a plain `MLP` under `PenaltyLoss`) has a zero gradient.
  `seed` fixes the collocation points and the initial network (torch's CPU generator, restored afterwards; the CUDA generators are not touched); `collocation(problem)` returns the points.
  The training runs on the CPU unless `accelerator` is `"gpu"` or `"auto"`, so the seed fixes the result on a machine with a given thread count.
  `Result.u` and `Result.hessian` (autograd) take numpy points of shape `(..., d)`; `Result.info` adds `stop`, `loss_terms`, `collocation` (`interior_points`, `boundary_points` and a SHA-256 fingerprint), `device`, `threads` (torch's CPU threads) and, for a network whose module carries a `report`, `network`; `history` holds the loss at the start of each step.
  The solve's callbacks get `on_step_end` with the loss terms after every step, so `MLflowLogger` logs a training curve.
  A neural solver supports a problem when its network, loss and sampler do (as for the composed classical solvers, #23).
- The neural parts, frozen dataclasses with the bases `Network` (`build(problem, collocation)` returns a torch module, which may carry a `report` mapping that the solver records as `Result.info["network"]`), `Loss` (`__call__(u, batch)` returns the loss terms, `"loss"` the total), `Sampler` (`sample(problem, rng)` returns a `Collocation`) and `Optimizer` (`build(parameters)`), in `maesolvers.neural`; `supports(problem)` is True by default on `Network` and `Sampler`, and on `Loss` True exactly for a `Dirichlet` problem:
  `MLP(width=10, depth=5, activation="sigmoid")`; `HardDirichlet(network, lift_width=20, distance_width=40, distance_points=300, fit_steps=10_000, fit_tol=1e-5)`, the ansatz `u = G + D N`, whose report gives, per fit of `G` and `D`, the iterations, the final largest gradient component, whether it converged, the mean squared error and the largest misfit, and the largest `|G - g|` and `|D|` at the boundary points; `PenaltyLoss(convexity_weight=1e4, boundary_weight=0.0)`, which raises `ValueError` with `boundary_weight > 0` on a batch without boundary points; `UniformSampler(interior_points=3000, boundary_points=400)`; `LBFGS(lr=1.0, history_size=100)` and `Adam(lr=1e-3)`.
  `Batch` (the collocation points and the data there, as tensors; `g`, the Dirichlet data, is None for another boundary condition), `Collocation` (read-only points, with `fingerprint()`) and `sample_interior(domain, n, rng)` are exported from `maesolvers.neural`; `maesolvers.neural.functional` holds the torch kernels `hessian`, `monge_ampere_residual`, `diagonal_convexity_penalty` and `mean_squared_misfit` (which raises `ValueError` on empty tensors).
  `NeuralSolver`, the parts and the preset are exported at the top level.
- `PenaltyMLP`, the penalty MLP of Nystrom-Vestberg 2023 as a preset with its parts' defaults (`width=10, depth=5, convexity_weight=1e4, interior_points=3000, boundary_points=400, fit_steps=10_000, fit_tol=1e-5, max_steps=5000`), in `SOLVERS`; 2D Dirichlet problems on `Box`, `Disk` and `CutQuarterDisk`.
- `ICNN(width=10, depth=5, init_std=1.0)` in `maesolvers.neural`, a network part: the input convex network of Amos et al. 2017 in the form of Nystrom-Vestberg 2023, eq. (3.7), softplus hidden layers, a linear output, every input feeding every layer, and hidden-to-hidden and hidden-to-output weights that are the elementwise squares of the trained parameters, so the network is convex in its input whatever the parameters (issue #13); 563 parameters at the defaults in 2D, all drawn at the start from the normal distribution with standard deviation `init_std`.
- `PenaltyLoss(equation_at_boundary=False)`: with True the equation term also sums over the boundary collocation points, as Nystrom-Vestberg 2023 do for their input convex network (Example 4.2); the convexity term stays a mean over the interior points, and a batch without boundary points raises `ValueError` at the first step.
- `Batch.f_boundary`, the right-hand side at the boundary collocation points, a required field after `g`: code that builds a `Batch` itself must pass it.
- `Batch.to(device)`, the batch with every tensor on the device and a None field kept None; the training loop moves the batch with it, so a new field needs no change there.
- `BFGS()` in `maesolvers.neural`, an optimizer part: one iteration of SciPy's `minimize(method="BFGS")` per training step, dense inverse Hessian approximation and SciPy's line search included, so it takes SciPy's iterates up to rounding; a failed line search leaves the parameters unchanged (the training stops as `"stalled"`), and it keeps `n^2` numbers for `n` parameters.
- `DirichletICNN`, the input convex network of Nystrom-Vestberg 2023 as a preset, in `SOLVERS` and exported at the top level with `ICNN` and `BFGS`: `NeuralSolver(network=ICNN(), loss=PenaltyLoss(convexity_weight=0, boundary_weight=100, equation_at_boundary=True), sampler=UniformSampler(), optimizer=BFGS(), max_steps=10_000)`, with the flat parameters `width`, `depth`, `init_std`, `boundary_weight` (positive: without an ansatz, the boundary term alone imposes `g`), `interior_points`, `boundary_points` and `max_steps`; Dirichlet problems in any dimension on `Box`, `Disk` and `CutQuarterDisk`.
- `BoundarySampling` (`sample_boundary(n, rng)`) and `BoundaryDistance` (`boundary_distance(x)`) in `maesolvers.core`: capabilities beside the closed `Domain` protocol, implemented by `Box` (any dimension), `Disk` and `CutQuarterDisk`.
- `maesolvers.core.nonnegative(name, value)`, the parameter check for a weight that 0 turns off.
- `maesolvers.core.boolean(name, value)`, the parameter check for a flag: a Python or numpy bool, returned as a plain `bool`; an int such as 1 raises `TypeError`.
- `maesolvers.core.check_part(owner, part, base)`, the check that a part passed to a solver or to another part is an instance of a well-defined subclass of its base (it was private to the classical iterations), for parts of your own that take parts.
- `maesolvers.metrics.summarize_errors(errors, threshold)`: `summarize` for errors measured elsewhere, such as in worker processes; it cannot see how a run stopped, so measure each seed with `seed_error`, which gives None for a run that stopped as `"diverged"`, and the two agree.
  `summarize` now states the rule for a run that stops at its step budget: a normal seed, not a failed one.
- `maesolvers.metrics.seed_error(result, key="max")`: one seed's error as `summarize` counts it, None for a failed seed (no result, a run that stopped as `"diverged"`, or a NaN or infinite error); `summarize` uses it, and the benchmark scripts pass its values to `summarize_errors`.
- The `neural-cpu` dependency group: the neural extra with torch from the PyTorch CPU index, used by CI and `just check`.
- The comparison table (issue #14): `benchmarks/run_comparison.py` runs every preset of `SOLVERS` with its defaults on the shared test set on a100server, the classical ones on the grid `n = 361`, the neural ones over seeds 0 to 9, and writes `benchmarks/comparison.csv`, each row with the preset's `hparams` and the machine; `benchmarks/comparison.py` (`just comparison-table`) writes README's tables of the max error, the non-convex fraction and the time from it, with notes derived from the rows, and a test checks that README is current and that the CSV covers every preset and problem with the presets' current `hparams`.
  A neural seed counts as a success when its max error is below 1% of the exact solution's oscillation `max u - min u` on the shared evaluation grid, which a constant added to `u` does not change.

### Changed

- `Newton` has the new field `start` (see Added), so the configuration and `hparams` of a bare `Newton` carry `"start": None`, and `tracking.flatten` logs it; the presets' `hparams` are unchanged.
- `summarize` counts a seed whose `Result.info["stop"]` is `"diverged"` (a neural run that stopped on a non-finite loss) as failed, through `seed_error`, with error `+inf`, though the network it returns, that of the step before, has a finite error (ADR 0007; it was left open to #14).
- `Result.info["stop"]` is part of the core contract (ADR 0007): when present it says why the run ended, and its value `"diverged"` marks a failed run for every solver, which `summarize` and `seed_error` count as failed; it was documented only for `NeuralSolver`, which sets it as before.
- `SOLVERS` lists the classical presets, then the neural ones, each in the order they were added: `FDNewton`, `FDPoisson`, `FDMonotone`, `FDHybrid`, `FDFiltered`, `PenaltyMLP`, `DirichletICNN`; the comparison table's columns follow it.
- `Standard9.residual` (and so `Standard9.jacobian` and `Result.info["residual_max"]`) multiplies each equation next to a curved boundary by the smallest step fraction of its stencil; on a box and at every node with a full stencil the factor is 1, so nothing changes there.
- `Standard9`'s `UnsupportedProblemError` message lists the four domains.
- `Standard9(spline_degree=k)` (and the presets) accept any `k` from 1 to 5 whatever `n`; on a box with fewer than `k` cells per side the spline takes degree `n`, so `FDNewton(n=2)` no longer raises.
- The fields of `Grid`, private to `Standard9`: `x` and `y` are gone; `h`, `row_scale`, `interpolate` and `interpolant` are new.
- The `neural` extra is torch and lightning >= 2 (it was torch alone).
- The `cli` extra is `jsonargparse[signatures]` (it was `jsonargparse`), which adds docstring-parser, so `maesolvers solve --solver.help <class>` shows each parameter's description from the numpy docstring next to its type and default.
- CI's `test` job installs the `neural-cpu` group and runs the neural tests on CPU; it fails if a CUDA wheel is installed.
- The development Python is 3.12 (`.python-version`): torch 2.7.1 has no wheel for 3.14.

### Fixed

- `tracking.run` starts its run in the experiment it names, by id: worker processes that opened runs in the same new experiment at once could log to `Default`, because `mlflow.set_experiment` does not make the experiment active when it loses the race to create it (#54).

## 0.0.2 (2026-10-22)

### Added

- `maesolvers.core`: `Box` (d-dimensional), `Disk`, `Ellipse` behind a `Domain` protocol with `dim`, `bbox`, `contains` and `interior`; the protocol is closed at 0.1, and further capabilities come as separate protocols.
- `Dirichlet(g)` and `Transport(target)` boundary condition types; for `Transport`, `Problem.f` is the source density and the target density is uniform 1 on `target`.
- The type aliases `Field` and `HessianField` (functions of points of shape `(..., d)`), `Observable` (a function of a solution to a number) and `BoundaryCondition` (`Dirichlet | Transport`) in `maesolvers.core`, for annotating problems and solvers of your own.
- `Problem.observables`: named numbers computed from a solution, reported by `evaluate`; each name must be an ASCII identifier (`ValueError` otherwise), so that it is also a valid MLflow metric name.
- `Result.hessian`, an optional callable Hessian, and `Result.hparams`, the solver's configuration as `class_path` and `init_args`, nested parts and containers of parts included.
  A `class_path` is `maesolvers.<Name>` for a class `maesolvers` exports (the form `--print_config` writes), so records and the MLflow parameters `tracking.flatten` makes from them do not depend on the internal module layout; for any other class it is the dotted module path.
- `Result.interior` and `Solution.interior`: an optional boolean mask of the method's interior nodes, where `evaluate` reads `node_hessian`; without it `evaluate` falls back to `problem.domain.interior(nodes)`, which may count nodes on a curved boundary as interior.
  `FDNewton` and `FDPoisson` set it.
- `Solver` base class with the fixed lifecycle `check -> setup -> run -> finalize`, the class variables `domains`, `boundaries` and `dims`, the instance method `supports(problem)` (a composed solver may override it; ADR 0003 does not yet say so, #23) and the `hparams` property; `Solution` and `Trace` are what its hooks return, both exported at the top level (ADR 0003).
  A subclass must carry its own `@dataclass(frozen=True)` decorator (`kw_only=True` is needed once it adds a field), else construction or `solve` raises `TypeError`; a constructor parameter that can be passed by position or is not a field (an `InitVar`) raises it too, so every solver takes keyword arguments only and `hparams` records all of them.
- `IterativeSolver` with the hooks `initial_guess(problem, context)`, `step(problem, context, iterate)` and `converged(problem, context, measure)`, the required class variable `measure`, and the shared loop over `max_iter`.
- `Callback` with the hooks `on_solve_start`, `on_step_end` and `on_solve_end`, passed as `solve(problem, callbacks=[...])`.
- `UnsupportedProblemError`, raised by `solve` for a problem outside the solver's declared support.
- `maesolvers.metrics.summarize(results, threshold, key="max")` and `Summary`: median, 5% and 95% quantiles and success rate of an `evaluate` field over seeds, a seed without a result (`None`) or with a NaN or infinite error counted as `+inf` (ADR 0007).
- `maesolvers.catalogue`: every problem under its T id, `CATALOGUE` (a read-only mapping from id to `Problem`) and `SHARED_TEST_SET`, the ids of the comparison table's rows; each problem is also exported at the top level (`from maesolvers import T1`).
  The shared test set T1, T2, T20, T8 (with `g = 0`), T12, T9, T10 follows the test-set specification and its point values; T3, T4, T5, T6, T7, T16 and T22 are catalogue-only, defined as their source papers print them (each `note` names the paper), except T4: its exact solution is the C^1 Alexandrov solution of its `f`, because the paper's `u` has a gradient jump that adds a singular measure to `f`.
  T7 has no exact solution and reports the observable `u_origin`, the value `u(0, 0)`.
  Every catalogue field raises `ValueError` for points whose last axis is not the problem's dimension.
- `CutQuarterDisk`, the domain of T6: the quarter of the unit disk in the first quadrant cut by `y > x - 1/2`.
- `maesolvers.classical`: solvers composed of a scheme and an iteration (ADR 0004).
  A scheme subclasses the abstract base `Scheme` (`supports`, `discretize`, `rhs`, `solution`) and is a frozen dataclass, so jsonargparse builds it from a `class_path` (ADR 0005).
  The scheme `Standard9(n, spline_degree)` declares `domains`, `boundaries` and `dims` on the class, discretises a problem (`discretize`) and evaluates the discrete equations on its grid (`rhs`, `second_derivatives`, `residual`, `jacobian`, `solve_laplace`, `solution`); the grid's fields are private to the scheme.
  The difference operators are the pure functions `difference_operators`, `monge_ampere` and `monge_ampere_jacobian` in `maesolvers.classical.schemes`; `Grid` there is what `Standard9.discretize` returns, and its fields may change with the scheme.
  An iteration subclasses `SchemeIteration(scheme, max_iter, tol)`, which asks the scheme for `supports`, the grid (`setup`) and the solution fields (`finalize`), and names in the class variable `requires` the scheme methods it calls; `check` raises a `TypeError` that names the missing ones, and construction raises a `TypeError` for a scheme that is a class or not a `Scheme`, or whose class fails the check a solver class passes (its own frozen dataclass decorator, keyword-only fields).
  The iterations are `Newton(scheme, max_iter=50, tol=1e-10)` and `PoissonIteration(scheme, max_iter=10_000, tol=1e-12)`.
  `Newton`, `PoissonIteration`, `Scheme`, `SchemeIteration` and `Standard9` are exported at the top level.
- `maesolvers.SOLVERS` (`maesolvers.registry`): the presets by name, read-only.
- `maesolvers.core.integer(name, value, minimum)` and `maesolvers.core.positive(name, value)`: the parameter checks the shipped solvers use, for the `__post_init__` of a solver of your own; they return a plain `int` or `float` and raise `TypeError` or `ValueError` with a message that names the parameter.
- `maesolvers.tracking.MLflowLogger`, a callback that logs a solve into the active MLflow run: git tags, the flattened `hparams` and the problem name as parameters at the start; every step's metrics at its step, the numeric fields of `evaluate` and `converged` at the end, after the timed part of the solve; `MLflowLogger.evaluation` keeps that `evaluate`.
  It raises `RuntimeError` when no run is active at the start or at the end of the solve.
  `tracking.flatten(config)` gives the dotted parameter names (`scheme.n`), and `tracking.git_tags()` the git state.
- `maesolvers.from_config(config) -> Solver`: builds a solver from its configuration, a mapping or a YAML or JSON file with `class_path` and `init_args`, nested parts in the same form (ADR 0005).
  This is the form `Result.hparams` records, so `from_config(result.hparams).solve(result.problem)` reruns a result; an `init_args` with no value means the defaults.
  A `class_path` is a class name maesolvers ships or the dotted path of any `Solver` subclass.
  A missing or unknown key or a wrong type raises `ValueError` naming the key, a value a constructor rejects raises what the constructor raises, and a missing file raises `FileNotFoundError`.
- The `maesolvers` console script (also `python -m maesolvers`) with the subcommand `solve`: `--config FILE` (keys `solver`, `problem` and `output`, a path relative to the file), `--solver` and dotted overrides such as `--solver.scheme.n 128`, `--problem ID`, and `--output FILE`.
  It prints `evaluate` as a table; `--output` writes `run` (a `--config` file that reruns the result), `metrics` and `info` as strict JSON, with numpy values as numbers and lists and NaN and infinity as `null`.
  It exits with status 2 for a wrong argument, an `--output` path that cannot be created, a value a constructor rejects or an unsupported problem, before any solve.
  Without the `cli` extra it exits with status 2 and a one-line message that names the extra and how to install it.
- The `cli` extra (jsonargparse >= 4.49), imported only by `from_config` and `maesolvers.cli`; `import maesolvers` imports no extra.
- `docs/guide.md`: solve a problem, pass parameters in the three ways, and write a solver of your own, with a worked example that `tests/test_guide.py` runs; `docs/methods/standard9.md` for the standard 9-point scheme and its two iterations.

### Changed

- `Problem(name, domain, f, boundary, exact=None, observables={}, note="")`: the boundary is a `Dirichlet` or `Transport` object; the fields `g`, `target` and the string `boundary` are gone.
- `Problem`, `Result` and the solvers are frozen dataclasses; `Problem` and `Result` compare by identity.
  The optional fields of `Problem` (after `boundary`), `Result` and `Solution` (after `u`) are keyword-only.
- `Result` stores read-only float copies of `nodes`, `values` and `node_hessian`, and `info` and `hparams` as deep copies in plain dicts.
- `Result.hessian` is now the callable Hessian; the discrete Hessian at the nodes moved to `Result.node_hessian`, and `nodes` and `values` are optional.
- `maesolvers.metrics.shared_grid(domain, size=None)` takes a domain instead of a problem and picks 201, 51 or 21 points per axis by dimension (`SHARED_GRID_SIZE`), instead of `shared_grid(problem, n=201)`.
- `evaluate(result)` takes the result only (the `n` parameter is gone; the shared grid's size follows the dimension) and reports `max`, `rms`, `max_grid`, `rms_grid`, `nonconvex_fraction`, `eigmin`, `eigmin_exact`, `time`, `iterations`, `unknowns` and the observables; errors on transport problems are taken after subtracting each side's mean.
  Convexity is taken at the interior nodes (`Result.interior`, else the nodes inside the domain) with the node Hessian, or on the shared grid with the callable Hessian, and a non-finite Hessian at any of these points gives NaN `nonconvex_fraction` and `eigmin`.
- `FDNewton` and `FDPoisson` take keyword arguments only, validate them on construction (`TypeError` for a float `n` or a string `tol`, numpy scalars stored as plain numbers), and raise `UnsupportedProblemError` instead of `TypeError` or `ValueError` for an unsupported problem.
- `FDNewton` reports a start that already meets `tol` as converged after one step instead of stalling.
- `FDNewton` and `FDPoisson` are presets: `FDNewton(n=64)` is `Newton(scheme=Standard9(n=64))` and `FDPoisson(n=64)` is `PoissonIteration(scheme=Standard9(n=64))`, with the same defaults, call style, numbers and class variables `domains`, `boundaries` and `dims` as before; their `hparams` stay flat and their `class_path` is `maesolvers.FDNewton` (`maesolvers.FDPoisson`), the form of every class maesolvers exports.
- `maesolvers.tracking` tags runs with the git checkout of the current working directory instead of the checkout that holds the package, and imports mlflow only when a run starts or a logger is used.
- `Result.info` holds `time`, `iterations`, `converged`, `history`, `unknowns`, `h` and `residual_max` for both finite-difference solvers.

### Removed

- `evaluate` no longer reports `rel_l2_nodes`, `rel_l2_grid`, `residual_rms`, `residual_max` and `eigmin_min`; `max_nodes` is now `max`.
- `maesolvers.metrics.solution_errors`, `residual` and `convexity`; `evaluate` is the one entry point.
- `boundary_points(n)` of `Box`, `Disk` and `Ellipse`: nothing in the library used it, and its `n` meant points per side for a box but points in total for a disk.
- The modules `maesolvers.domains` and `maesolvers.result`; their classes are in `maesolvers.core` and exported at the top level.
- The module `maesolvers.solvers` and its submodules `fd_newton`, `fd_poisson` and `_grid`; import the solvers from `maesolvers` or `maesolvers.classical`.
- The bare `Solver` protocol of `maesolvers.solvers.base`, and `Result.info["solver"]`, `["residual_history"]`, `["update_history"]` and `["stalled"]`.
- The catalogue `P1` to `P5` and the module `maesolvers.problems`, with no aliases (ADR 0007): P4 is now T1, P5 is T2, P1 is T8 with `g = 0` (its solution `(|x|^2 - 1)/2` differs by the constant 1/2), P3 is T12, and P2 left the catalogue.
