# Every solver has one lifecycle, modelled on LightningModule, in plain numpy

Every solver subclasses `Solver`, whose `solve(problem) -> Result` is fixed: `check -> setup -> run -> finalize`.
A method author writes hooks only; timing, iteration history, convergence bookkeeping, hyperparameter capture and the construction of the `Result` live in the base class, so every method reports the same fields.
Two family bases fill in `run`: `IterativeSolver` (hooks `initial_guess`, `step`, `converged`) for the classical iterations, and `NeuralSolver`, which builds a `LightningModule` internally and hands it to a Lightning `Trainer`.
Each solver declares on the class which domains and boundary conditions it `supports`, so table scripts skip unsupported pairs instead of catching errors.
Callbacks (`on_solve_start`, `on_step_end`, `on_solve_end`) are the same protocol for both families; `NeuralSolver` forwards them to Lightning callbacks.

## Considered options

- **Every solver is a `LightningModule`** (solo-learn's shape): rejected, because classical solvers have no training loop, and `FDPoisson(n=64).solve(T1)` would then need torch.
- **A bare `solve()` protocol with no base class** (0.0.x): rejected, because each method then re-implements timing, history and result fields, and they drift apart.
- **Lightning only for the neural family, with its own unrelated API**: rejected, because users would learn two interfaces; Lightning stays an implementation detail of `NeuralSolver`.

## Consequences

The `Solver` protocol still holds for third-party code: any object with `solve(problem) -> Result` works with `evaluate`, but only subclasses of `Solver` get the shared bookkeeping, the config support and the callbacks.
