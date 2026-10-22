# Solvers are compositions of parts, with presets for the common ones

A classical solver is a scheme (grid, difference operators, residual, Jacobian) plus an iteration, for example `Newton(scheme=Standard9(n=64))` or `SemismoothNewton(scheme=Monotone17(n=64))`.
A neural solver is a network (MLP, ICNN), a loss (penalty, QS), a collocation sampler and an optimizer, run by the shared training loop.
The reason is that the parts vary independently and the literature compares them that way: the standard 9-point scheme picks a non-convex root under Newton but the convex one under the Poisson iteration (`benchmarks/standard_fd.md`, Finding 2), and ICNN differs from the penalty MLP only in its network.
Presets name the common pairs (`FDPoisson` is `Standard9` with `PoissonIteration`) and are what the `SOLVERS` registry and the comparison tables list, so the simple call `FDPoisson(n=64).solve(T1)` stays.
Difference operators and losses are pure functions, testable without a solver.
Not every pair is valid; an iteration checks in `check` that its scheme provides what it needs (for example, a Jacobian).
