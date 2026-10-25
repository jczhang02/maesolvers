# Neural methods share a training loop and an evaluation, not hyperparameters

Resolves issue #4.
One fixed protocol (same optimizer, schedule and step budget for all methods) can fail for a particular method: an optimizer that works for the penalty MLP may stall for a convex architecture.
So the neural family shares what makes results comparable, and each method keeps what makes it train:

| Shared by all neural methods | Set per method (defaults from the source paper) |
|---|---|
| the training loop (`NeuralSolver` on a Lightning `Trainer`) | network architecture and size |
| the collocation point set of each sampler and seed, regenerated from them and identified by a fingerprint | loss terms and their weights |
| seeds $0$ to $9$ | optimizer, learning-rate schedule, step budget |
| the evaluation grid and the metrics (ADR 0007) | adaptive resampling, reported as a separate variant |
| float64 (`precision="64-true"`) | |
| the names of logged metrics | |

The collocation set is per sampler and seed: it is regenerated deterministically from the pair (sampler, seed), so two methods that share a sampler and a seed share the points, and it is identified by the SHA-256 fingerprint recorded in `Result.info["collocation"]`.
It is not persisted; a sampler that reads a stored set can be added later without changing a signature (issue #52).

"One shared training loop" in the v0.1 plan means shared code, not shared hyperparameters.
Each method's settings are recorded in its `Result` and in MLflow (ADR 0005), and its documentation states where they come from.
How failed seeds are counted is in ADR 0007 (`summarize`).
