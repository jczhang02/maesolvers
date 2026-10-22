# Agent Instructions

Python library of solvers for the Monge-Ampere equation `det D^2 u = f`, classical and neural, behind one interface and measured on one catalogue of test problems.
`README.md` maps the directories and the public interface.
`AGENTS.md` is a symlink to this file, so Codex reads the same rules: edit `CLAUDE.md` only.

## Python

- One package, `maesolvers` (`src/maesolvers/`), Python 3.10+, numpy and scipy only.
  torch arrives with the `neural` extra; nothing in 0.0.x imports it.
  mlflow lives in the `experiments` dependency group, used only by `maesolvers.tracking`.
- `uv sync --group dev` installs the dev tools.
  `just check` runs what CI runs (`ruff format --check`, `ruff check`, `pytest`); CI repeats it on Python 3.10 and 3.13.
  Run it before you commit.
- `pyproject.toml` sets the line rules: the formatter wraps at 88 columns, E501 fails a line over 100.
- Markdown: one sentence per line, no fixed width, so a diff shows only the sentences that changed.
- `just` lists the common commands (`justfile`).

## Interface

- A `Problem` (domain, right-hand side, boundary condition, exact solution when known), any object with `solve(problem) -> Result`, and `evaluate(result)` in `maesolvers.metrics`.
  A new solver is one class; it plugs into the same `evaluate` and the same comparison table.
- Catalogue problems `P1` to `P5` are provisional.
  A number, once used, is never reused, so tables from different methods line up.
  The numbering will be aligned with the research repository (`jczhang02/monge-ampere-equations`) when it fixes its test set.

## Reproduction

- A method counts as reproduced when it matches the source paper's own table, on the paper's own test problem, to the printed digits.
- `benchmarks/sources.md` transcribes the published tables with their conventions; `benchmarks/standard_fd.md` compares them with ours; the CSVs and `plot_standard_fd.py` next to them draw the README figure.
- The papers' grid convention is `h = 1/N`; none of them states it.
- A mismatch is recorded as a finding in the comparison file, not tuned away.
- `tests/test_reproduction.py` pins the published numbers; change a pinned value only together with the finding that justifies it.

## Experiments

- Run numerical experiments and training on `a100server` (ssh host alias), not on the local machine.
- Commit first, then `just push` and run in the server checkout with `just remote <cmd>`, or `just remote-bg <name> <cmd>` for long runs (tmux session `maesolvers-<name>`, log in the data dir).
  After a change to `uv.lock`, run `just server-sync`: the server reaches PyPI only through the Shanhe mirror.
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
- Ask first: commit, push, open or merge a PR, push a tag or create a release, amend or rebase pushed commits.
- Never: force-push `main`; commit `out/`, `dist/`, secrets, or large data (it belongs in MLflow or the a100server data dir); `reset --hard`, `clean -f`, `checkout .` unless JC asks.

## Agent skills

### Issue tracker

Issues are tracked as GitHub Issues on `jczhang02/maesolvers` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels are used as-is (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
