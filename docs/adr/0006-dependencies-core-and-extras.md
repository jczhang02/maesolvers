# The core depends on numpy and scipy only; each method family is an extra

`import maesolvers` and every classical finite-difference solver need numpy and scipy and nothing else.
Each further family brings its dependencies as an optional extra, imported only inside that family's subpackage and only when it is used: `neural` (torch, lightning >= 2) for `maesolvers.neural`, `fem` (scikit-fem) for the finite-element family from v0.3, and `cli` (jsonargparse) for configs and the command line.
mlflow stays in the `experiments` dependency group, outside the library.
CI runs one job with no extras, which imports the package and runs every classical solver, and one job with the `neural` extra on CPU.
This replaces the rule "numpy and scipy only", which the accepted roadmap (Lightning in v0.1, scikit-fem in v0.3) had already broken.
