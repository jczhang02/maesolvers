# Changelog

Every change to the public interface is listed here; 0.x may break the interface, but never silently.

## Unreleased (0.0.2)

### Added

- `maesolvers.core`: `Box` (d-dimensional), `Disk`, `Ellipse` behind a `Domain` protocol with `dim`, `bbox`, `contains` and `interior`; the protocol is closed at 0.1, and further capabilities come as separate protocols.
- `Dirichlet(g)` and `Transport(target)` boundary condition types; for `Transport`, `Problem.f` is the source density and the target density is uniform 1 on `target`.
- `Problem.observables`: named numbers computed from a solution, reported by `evaluate`.
- `Result.hessian`, an optional callable Hessian, and `Result.hparams`, the solver's configuration as `class_path` and `init_args`, nested parts and containers of parts included.
- `Result.interior` and `Solution.interior`: an optional boolean mask of the method's interior nodes, where `evaluate` reads `node_hessian`; without it `evaluate` falls back to `problem.domain.interior(nodes)`, which may count nodes on a curved boundary as interior.
  `FDNewton` and `FDPoisson` set it.
- `Solver` base class with the fixed lifecycle `check -> setup -> run -> finalize`, the class variables `domains`, `boundaries` and `dims`, the instance method `supports(problem)` (a composed solver may override it) and the `hparams` property; `Solution` and `Trace` are what its hooks return, both exported at the top level (ADR 0003).
  A subclass must carry its own `@dataclass(frozen=True)` decorator, else construction or `solve` raises `TypeError`.
- `IterativeSolver` with the hooks `initial_guess(problem, context)`, `step(problem, context, iterate)` and `converged(problem, context, measure)`, the required class variable `measure`, and the shared loop over `max_iter`.
- `Callback` with the hooks `on_solve_start`, `on_step_end` and `on_solve_end`, passed as `solve(problem, callbacks=[...])`.
- `UnsupportedProblemError`, raised by `solve` for a problem outside the solver's declared support.
- `maesolvers.metrics.summarize(results, threshold, key="max")` and `Summary`: median, 5% and 95% quantiles and success rate of an `evaluate` field over seeds, a seed without a result (`None`) or with a NaN or infinite error counted as `+inf` (ADR 0007).
- `maesolvers.metrics.shared_grid(domain, size=None)` takes a domain and picks 201, 51 or 21 points per axis by dimension.
- `maesolvers.catalogue`: every problem under its T id, `CATALOGUE` (a read-only mapping from id to `Problem`) and `SHARED_TEST_SET`, the ids of the comparison table's rows; each problem is also exported at the top level (`from maesolvers import T1`).
  The shared test set T1, T2, T20, T8 (with `g = 0`), T12, T9, T10 follows the test-set specification and its point values; T3, T4, T5, T6, T7, T16 and T22 are catalogue-only, defined as their source papers print them (each `note` names the paper).
  T7 has no exact solution and reports the observable `u(0,0)`.
- `CutQuarterDisk`, the domain of T6: the quarter of the unit disk in the first quadrant cut by `y > x - 1/2`.

### Changed

- `Problem(name, domain, f, boundary, exact=None, observables={}, note="")`: the boundary is a `Dirichlet` or `Transport` object; the fields `g`, `target` and the string `boundary` are gone.
- `Problem`, `Result` and the solvers are frozen dataclasses; `Problem` and `Result` compare by identity.
  The optional fields of `Problem` (after `boundary`), `Result` and `Solution` (after `u`) are keyword-only.
- `Result` stores read-only float copies of `nodes`, `values` and `node_hessian`, and `info` and `hparams` as deep copies in plain dicts.
- `Result.hessian` is now the callable Hessian; the discrete Hessian at the nodes moved to `Result.node_hessian`, and `nodes` and `values` are optional.
- `evaluate(result)` reports `max`, `rms`, `max_grid`, `rms_grid`, `nonconvex_fraction`, `eigmin`, `eigmin_exact`, `time`, `iterations`, `unknowns` and the observables; errors on transport problems are taken after subtracting each side's mean.
  Convexity is taken at the interior nodes (`Result.interior`, else the nodes inside the domain) with the node Hessian, or on the shared grid with the callable Hessian, and a non-finite Hessian at any of these points gives NaN `nonconvex_fraction` and `eigmin`.
- `FDNewton` and `FDPoisson` take keyword arguments only, validate them on construction (`TypeError` for a float `n` or a string `tol`, numpy scalars stored as plain numbers), and raise `UnsupportedProblemError` instead of `TypeError` or `ValueError` for an unsupported problem.
- `FDNewton` reports a start that already meets `tol` as converged after one step instead of stalling.
- `Result.info` holds `time`, `iterations`, `converged`, `history`, `unknowns`, `h` and `residual_max` for both finite-difference solvers.

### Removed

- `evaluate` no longer reports `rel_l2_nodes`, `rel_l2_grid`, `residual_rms`, `residual_max` and `eigmin_min`; `max_nodes` is now `max`.
- `maesolvers.metrics.solution_errors`, `residual` and `convexity`; `evaluate` is the one entry point.
- `boundary_points(n)` of `Box`, `Disk` and `Ellipse`: nothing in the library used it, and its `n` meant points per side for a box but points in total for a disk.
- The bare `Solver` protocol of `maesolvers.solvers.base`, and `Result.info["solver"]`, `["residual_history"]`, `["update_history"]` and `["stalled"]`.
- The catalogue `P1` to `P5` and the module `maesolvers.problems`, with no aliases (ADR 0007): P4 is now T1, P5 is T2, P1 is T8 with `g = 0` (its solution `(|x|^2 - 1)/2` differs by the constant 1/2), P3 is T12, and P2 left the catalogue.
