# Parameters come from constructor signatures, through jsonargparse, not Hydra

A solver is built in one of three ways, modelled on Lightning: Python keyword arguments, a dict or YAML file through `maesolvers.from_config`, or the command line (`maesolvers solve --config exp.yaml --solver.scheme.n 128 --problem T2`).
All three are derived from the constructors' type-annotated signatures by jsonargparse, the parser under LightningCLI, with nested parts written as `class_path` / `init_args`; a misspelt key is an error.
Every `Result` records the full hyperparameters, and the MLflow callback logs the same record, so any result can be rerun from its config.
jsonargparse is in the `cli` extra; the core never imports it.

## Considered options

- **Hydra with `_target_`** (CAMO's shape): rejected, because defaults would live in YAML files apart from the code and drift from it, and the library would not be usable without the config tree.
  Hydra's multirun is the one feature we lose; sweeps are Python scripts under `experiments/`.
- **The constructor takes the whole config** (solo-learn's shape): rejected, because a solver could then not be built from plain keyword arguments.
