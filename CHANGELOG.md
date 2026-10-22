# Changelog

Every change to the public interface is listed here; 0.x may break the interface, but never silently.

## Unreleased (0.0.2)

### Added

- `maesolvers.core`: `Box` (d-dimensional), `Disk`, `Ellipse` behind a `Domain` protocol with `dim`, `bbox`, `contains` and `interior`.
- `Dirichlet(g)` and `Transport(target)` boundary condition types.
- `Problem.observables`: named numbers computed from a solution, reported by `evaluate`.
- `Result.hessian`, an optional callable Hessian, and `Result.hparams`, the solver's configuration as `class_path` and `init_args`.
- `Solver` base class with the fixed lifecycle `check -> setup -> run -> finalize`, the class variables `domains`, `boundaries` and `dims`, the predicate `supports(problem)` and the `hparams` property; `Solution` and `Trace` are what its hooks return (ADR 0003).
- `IterativeSolver` with the hooks `initial_guess`, `step` and `converged` and the shared loop over `max_iter`.
- `Callback` with the hooks `on_solve_start`, `on_step_end` and `on_solve_end`, passed as `solve(problem, callbacks=[...])`.
- `UnsupportedProblemError`, raised by `solve` for a problem outside the solver's declared support.
- `maesolvers.metrics.summarize(metrics, threshold, key="max")` and `Summary`: median, 5% and 95% quantiles and success rate over seeds, failed seeds counted as `+inf` (ADR 0007).
- `maesolvers.metrics.shared_grid(domain, size=None)` takes a domain and picks 201, 51 or 21 points per axis by dimension.

### Changed

- `Problem(name, domain, f, boundary, exact=None, observables={}, note="")`: the boundary is a `Dirichlet` or `Transport` object; the fields `g`, `target` and the string `boundary` are gone.
- `Problem`, `Result` and the solvers are frozen dataclasses; `Problem` and `Result` compare by identity.
- `Result.hessian` is now the callable Hessian; the discrete Hessian at the nodes moved to `Result.node_hessian`, and `nodes` and `values` are optional.
- `evaluate(result)` reports `max`, `rms`, `max_grid`, `rms_grid`, `nonconvex_fraction`, `eigmin`, `eigmin_exact`, `time`, `iterations`, `unknowns` and the observables; errors on transport problems are taken after subtracting each side's mean.
- `FDNewton` and `FDPoisson` take keyword arguments only, validate them on construction, and raise `UnsupportedProblemError` instead of `TypeError` or `ValueError` for an unsupported problem.
- `Result.info` holds `time`, `iterations`, `converged`, `history`, `unknowns`, `h` and `residual_max` for both finite-difference solvers.

### Removed

- `evaluate` no longer reports `rel_l2_nodes`, `rel_l2_grid`, `residual_rms`, `residual_max` and `eigmin_min`; `max_nodes` is now `max`.
- `maesolvers.metrics.solution_errors`, `residual` and `convexity`; `evaluate` is the one entry point.
- The bare `Solver` protocol of `maesolvers.solvers.base`, and `Result.info["solver"]`, `["residual_history"]`, `["update_history"]` and `["stalled"]`.
