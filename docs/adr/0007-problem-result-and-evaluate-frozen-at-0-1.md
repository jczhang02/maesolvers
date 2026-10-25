# Problem, Result and evaluate are frozen at 0.1, designed for v0.2 now

The research code (`src/mae` in jczhang02/monge-ampere-equations) imports the catalogue and the metrics from maesolvers once 0.1 is out, so 0.1 freezes `Problem`, `Result` and `evaluate`.
They are therefore shaped now for what v0.2 needs, even though 0.1 ships two-dimensional problems only.

## Problem

- `Box` is $d$-dimensional: `lo` and `hi` are tuples of length $d$, and points are arrays of shape `(..., d)`.
- A boundary condition is a typed object, `Dirichlet(g)` or `Transport(target)`, not a string with two optional fields.
- `observables` holds named quantities for problems without an exact solution (for T7, $u(0, 0)$, the number the literature prints).
- The catalogue uses the T ids of the shared test-set specification (`docs/test-set.md` in the research repository), with its definitions and point-value tests.
  P1 to P5 are retired, with no aliases (pre-alpha), and their numbers are not reused: P4 is T1, P5 is T2, P1 is T8 with $g = 0$, P3 is T12, and P2 leaves the catalogue.

## Result

A `Result` carries the solution as a callable `u(points)`, and optionally `hessian(points)`, a callable Hessian (autograd for neural methods).
A grid method also carries `nodes`, its `values` there, and its own discrete Hessian at the nodes, and optionally `interior`, a boolean mask of the nodes that are interior; without it, `evaluate` asks the domain.
`Result.info["stop"]` records why a solve stopped (such as `"converged"` or `"max_steps"`); the value `"diverged"` is shared by every solver and marks a failed run.

## evaluate

The primary points are the method's `nodes` when it has them, otherwise the shared evaluation grid (the bounding-box grid, points in the domain only: $201 \times 201$ in 2D; $51^3$ and $21^4$ provisional).
This is the literature's practice: classical methods report at their own nodes, neural methods never at their training points.
The fields are:

- `max`, `rms`: the max absolute and RMS error at the primary points;
- `max_grid`, `rms_grid`: the same on the shared grid through the method's interpolant, a secondary value for grid methods, never used for ranking;
- `nonconvex_fraction`, the share of points with a negative smallest Hessian eigenvalue, at the interior nodes (discrete Hessian) or on the shared grid (callable Hessian); `eigmin` and the exact solution's `eigmin` on the same points go to appendix tables;
- `time`, `iterations`, `unknowns`;
- the problem's `observables`, evaluated on the solution.

Errors for T12 (transport, solution unique up to a constant) are computed after subtracting each side's mean over the points.
The relative $L^2$ errors of 0.0.x are dropped, because no paper prints them.

`summarize(results, threshold)` aggregates the seeds of one method on one problem: median, $5$% and $95$% quantiles, and success rate.
`results` is a `Sequence[Result | None]`, where `None` is a seed that has no result (a timeout or a crash).
A failed seed (divergence, NaN, timeout) enters the quantiles with error $+\infty$ and is never dropped, so a method that fails on more than half the seeds has an infinite median, printed as "failed".
The success threshold is a parameter, with no default in 0.1.
