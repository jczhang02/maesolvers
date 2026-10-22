# Changelog

Every change to the public interface is listed here; 0.x may break the interface, but never silently.

## Unreleased (0.0.2)

### Added

- `maesolvers.core`: `Box` (d-dimensional), `Disk`, `Ellipse` behind a `Domain` protocol with `dim`, `bbox`, `contains` and `interior`; the protocol is closed at 0.1, and further capabilities come as separate protocols.
- `Dirichlet(g)` and `Transport(target)` boundary condition types; for `Transport`, `Problem.f` is the source density and the target density is uniform 1 on `target`.
- The type aliases `Field` and `HessianField` (functions of points of shape `(..., d)`), `Observable` (a function of a solution to a number) and `BoundaryCondition` (`Dirichlet | Transport`) in `maesolvers.core`, for annotating problems and solvers of your own.
- `Problem.observables`: named numbers computed from a solution, reported by `evaluate`; each name must be an ASCII identifier (`ValueError` otherwise), so that it is also a valid MLflow metric name.
- `Result.hessian`, an optional callable Hessian, and `Result.hparams`, the solver's configuration as `class_path` and `init_args`, nested parts and containers of parts included.
  A `class_path` is `maesolvers.<Name>` for a class `maesolvers` exports (the form `--print_config` writes), so records and the MLflow parameters `tracking.flatten` makes from them do not depend on the internal module layout; for any other class it is the dotted module path.
- `Result.interior` and `Solution.interior`: an optional boolean mask of the method's interior nodes, where `evaluate` reads `node_hessian`; without it `evaluate` falls back to `problem.domain.interior(nodes)`, which may count nodes on a curved boundary as interior.
  `FDNewton` and `FDPoisson` set it.
- `Solver` base class with the fixed lifecycle `check -> setup -> run -> finalize`, the class variables `domains`, `boundaries` and `dims`, the instance method `supports(problem)` (a composed solver may override it) and the `hparams` property; `Solution` and `Trace` are what its hooks return, both exported at the top level (ADR 0003).
  A subclass must carry its own `@dataclass(frozen=True)` decorator, else construction or `solve` raises `TypeError`.
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
  An iteration subclasses `SchemeIteration(scheme, max_iter, tol)`, which asks the scheme for `supports`, the grid (`setup`) and the solution fields (`finalize`), and names in the class variable `requires` the scheme methods it calls; `check` raises a `TypeError` that names the missing ones, and construction raises a `TypeError` for a scheme that is a class, not a `Scheme`, or not a frozen dataclass.
  The iterations are `Newton(scheme, max_iter=50, tol=1e-10)` and `PoissonIteration(scheme, max_iter=10_000, tol=1e-12)`.
  `Newton`, `PoissonIteration`, `Scheme`, `SchemeIteration` and `Standard9` are exported at the top level.
- `maesolvers.SOLVERS` (`maesolvers.registry`): the presets by name, read-only.
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
- `FDNewton` and `FDPoisson` are presets: `FDNewton(n=64)` is `Newton(scheme=Standard9(n=64))` and `FDPoisson(n=64)` is `PoissonIteration(scheme=Standard9(n=64))`, with the same defaults, call style, numbers and class variables `domains`, `boundaries` and `dims` as before; their `hparams` stay flat and their `class_path` is now `maesolvers.classical.presets.FDNewton` (`.FDPoisson`).
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
