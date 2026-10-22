# Agent Instructions

Python library of solvers for the Monge-Ampere equation `det D^2 u = f`, classical and neural, behind one interface and measured on one catalogue of test problems.
`README.md` maps the directories and the public interface.
`AGENTS.md` is a symlink to this file, so Codex reads the same rules: edit `CLAUDE.md` only.

## Python

- One package, `maesolvers` (`src/maesolvers/`), Python 3.10+.
  The core and every classical solver need numpy and scipy only; each further family is an extra imported only inside its subpackage (`neural`: torch and lightning, `fem`: scikit-fem, `cli`: jsonargparse), per ADR 0006.
  mlflow lives in the `experiments` dependency group, used only by `maesolvers.tracking`.
- Public classes and functions carry type annotations and numpy-style docstrings.
- `uv sync --group dev` installs the dev tools.
  `just check` runs what CI runs (its recipe in `justfile` lists the steps); CI repeats it on Python 3.10 and 3.13.
  Run it before you commit.
- `pyproject.toml` sets the line rules: the formatter wraps at 88 columns, E501 fails a line over 100.
- Markdown: one sentence per line, no fixed width, so a diff shows only the sentences that changed.
- `just` lists the common commands (`justfile`).

## Interface

- A `Problem`, a solver with `solve(problem) -> Result`, and `evaluate(result)` in `maesolvers.metrics`.
- The design for 0.1 is fixed in `docs/adr/` 0003 to 0008; read the ADR before you change a solver, `Problem`, `Result`, `evaluate` or the training loop:
  - one solver lifecycle modelled on LightningModule, in plain numpy (0003);
  - solvers composed of a scheme and an iteration, or a network, a loss and a sampler, with presets (0004);
  - parameters from constructor signatures through jsonargparse (0005);
  - `Problem`, `Result` and `evaluate` frozen at 0.1 (0007);
  - the neural training protocol (0008).
  Known deviations, each with its issue: composed solvers (`Newton`, `PoissonIteration`) answer `supports(problem)` on the instance, not through class variables (ADR 0003, #23; do not add class-level `domains` to them); the `neural` extra lacks `lightning>=2` and CI has no CPU neural job (ADR 0006, #12).
- Problems carry the T ids of the shared test-set specification (`docs/test-set.md` in `jczhang02/monge-ampere-equations`); an id once used is never reused, so tables from different methods line up.

## Reproduction

- A reproduction is judged against the source paper's own table, on the paper's own test problem, at one of the five levels of ADR 0002 (exact, order of magnitude, statistical, figure only, mismatch); `benchmarks/<method>.md` states the level.
- `benchmarks/sources.md` transcribes the published tables with their conventions; `benchmarks/standard_fd.md` compares them with ours; the CSVs and `plot_standard_fd.py` next to them draw the README figure.
- The papers' grid convention is `h = 1/N`, `N` cells per side; none of them states it (ADR 0001).
- A mismatch is recorded as a finding in the comparison file, not tuned away.
- `tests/test_reproduction.py` pins the published numbers; change a pinned value only together with the finding that justifies it.

## Experiments

- Run numerical experiments and training on `a100server` (ssh host alias), not on the local machine.
- Commit first, then `just push` and run in the branch's own server worktree (`/jczhang02/maesolvers-wt/<branch>`) with `just remote <cmd>`, or `just remote-bg <name> <cmd>` for long runs (tmux session `maesolvers-<name>`, log in the data dir).
  After a change to `uv.lock`, run `just server-sync` (it installs every extra into the shared venv, under a lock): the server reaches PyPI only through the Shanhe mirror.
  Pick a free GPU with `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`; other runs may share the server.
- Log every experiment to MLflow with `maesolvers.tracking.run`; it tags the git commit and whether the checkout was dirty.
  The server is shared with the research repository; name experiments `maesolvers-<method>-<problem>`.
  The UI is at `just mlflow-ui` (port 30000); it is started from the research repository, not from here.
- Reproduction receipts (the CSVs and the comparison under `benchmarks/`) stay in the repo.
  Everything else an experiment produces (training metrics, checkpoints, sweeps, figures) goes to MLflow, not into the repo.
  External input data goes to `/jczhang02/maesolvers-data/data/` on a100server.

## Git

- Always: work in a git worktree under `.worktrees/`; never `git switch` or `git checkout` a branch in the main working tree: `git worktree add .worktrees/<branch-with-dashes> -b <branch> main`, and `git worktree remove` it after the merge.
  Conventional Commits: fix a message the commit-msg hook rejects; the hook always runs.
  Stage only the files you changed (`git add <path>`, not `-A`), because several agent sessions may share a working tree.
  Run `just check` and look at `git status` for untracked inputs before you commit.
- Free to do without asking: branch in a worktree, commit, push a branch, open a PR, run experiments on a100server, create and update issues.
- Ask first: merge into `main`, push a tag or create a release, amend or rebase pushed commits, change a pinned reproduction value, change an accepted ADR.
- Chains of dependent PRs use `gh stack`.
- Never: force-push `main`; commit `out/`, `dist/`, secrets, or large data (it belongs in MLflow or the a100server data dir); `reset --hard`, `clean -f`, `checkout .` unless JC asks.

## Documentation

- No docs site: `README.md`, `docs/guide.md` (solve a problem, pass parameters, plug in your own solver), and one `docs/methods/<name>.md` per method, or per scheme with its iterations as `standard9.md` (the mathematics, the source paper, the parameters, the known limits, a link to the `benchmarks/` file that holds its reproduction).
- Every change to the public interface gets a line in `CHANGELOG.md`; 0.x may break the interface, but never silently.

## Agent skills

### Issue tracker

Issues are tracked as GitHub Issues on `jczhang02/maesolvers` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels are used as-is (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
