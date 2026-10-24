"""Write the README comparison table from comparison.csv, or check that it is current.

comparison.csv holds one row per run of a preset of ``SOLVERS`` on a problem of
the shared test set (``run_comparison.py`` writes it on a100server): one row
for a classical preset, one per seed for a neural one, and one row with status
``unsupported`` for a pair the preset does not support. This module defines the
receipt (its columns, the grid, the seeds, the options each preset takes, and
how rerun rows merge into it) and turns the rows into three tables, the max
error, the non-convex fraction and the time, with the problems as rows and the
presets as columns in the order of ``SOLVERS``, and their notes, which it
derives from the rows. It writes them into README.md between the two marker
lines. It also holds :func:`with_parts`, which the diagnostic runs of the
neural receipts use to swap one part of a preset. It needs numpy and
maesolvers only, so its tests
(tests/test_comparison.py) run without extras. From the repository root::

    python benchmarks/comparison.py          # rewrite the README block
    python benchmarks/comparison.py --check  # exit 1 if README is not current
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, get_args, get_type_hints

import numpy as np

from maesolvers import CATALOGUE, SOLVERS, Monotone17, Problem, QuadraticForm, Solver
from maesolvers.catalogue import SHARED_TEST_SET
from maesolvers.classical.monotone import stencil_dimension
from maesolvers.metrics import shared_grid, summarize_errors
from maesolvers.neural import NeuralSolver

HERE = Path(__file__).parent
CSV = HERE / "comparison.csv"
README = HERE.parent / "README.md"

N = 361
"""The classical grid: the largest N the papers print in 2D."""
SEEDS = tuple(range(10))
"""The neural seeds (ADR 0008)."""

START = "<!-- comparison table: written by benchmarks/comparison.py, do not edit -->"
END = "<!-- comparison table end -->"

FIELDS = (
    "solver",
    "problem",
    "seed",
    "n",
    "status",
    "reason",
    "max",
    "rms",
    "max_grid",
    "rms_grid",
    "nonconvex_fraction",
    "eigmin",
    "eigmin_exact",
    "time",
    "iterations",
    "unknowns",
    "converged",
    "stop",
    "device",
    "threads",
    "cpu",
    "commit",
    "hparams",
)
"""The columns of comparison.csv."""

STATUSES = ("ok", "failed", "unsupported")
"""``ok``: a finite error; ``failed``: an exception, or a failed seed by the rule
of :func:`~maesolvers.metrics.seed_error` (a non-finite error, or a neural run
that stopped as ``"diverged"``; ADR 0007); ``unsupported``: the preset does not
support the problem, and ``reason`` says why."""

RELATIVE_THRESHOLD = 1e-2
"""A neural seed succeeds when its max error is below this share of the
oscillation ``max u - min u`` of the exact solution on the shared evaluation
grid, which a constant added to ``u`` does not change."""


PLANNED = {
    "2D only": "n-dimensional boxes",
}
"""The unsupported reasons that the roadmap's 0.2 still removes for the
finite-difference presets, by the feature that removes them (README, Roadmap;
the scope of issue #14). Transport boundary conditions came to them as one
preset of its own, ``FDTransport``, so the other presets keep ``Dirichlet
only``. The neural presets keep their reasons: 0.2 plans no neural transport
preset, and ``PenaltyMLP`` stays 2D only, as its penalty makes the network
convex only in 2D."""


def preset_options(name: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """Those of ``options`` that the constructor of the preset ``name`` takes.

    The runner's options (the grid ``n``, the neural ``max_steps``, the
    ``seed``) go to every preset whose signature has them (ADR 0005), and
    only there; an option of value None is left out.
    """
    parameters = inspect.signature(SOLVERS[name]).parameters
    return {k: v for k, v in options.items() if k in parameters and v is not None}


def with_parts(solver: NeuralSolver, **parts: Any) -> NeuralSolver:
    """A :class:`NeuralSolver` with the parts and settings of ``solver``, but ``parts``.

    A preset's parts are fields with ``init=False``, so
    ``dataclasses.replace(preset, optimizer=BFGS())`` cannot swap one; this
    copies every field of :class:`NeuralSolver` from ``solver`` and overrides
    those named in ``parts`` (``optimizer=BFGS()``, say).

    Parameters
    ----------
    solver
        The neural solver to copy, usually a preset.
    **parts
        The fields to replace, by name.

    Returns
    -------
    NeuralSolver

    Raises
    ------
    TypeError
        For a name in ``parts`` that is not a field of :class:`NeuralSolver`.
    """
    fields = {f.name for f in dataclasses.fields(NeuralSolver) if f.init}
    if unknown := sorted(set(parts) - fields):
        raise TypeError(f"not fields of NeuralSolver: {', '.join(unknown)}")
    return NeuralSolver(**{name: getattr(solver, name) for name in fields} | parts)


def unsupported_reason(solver: Solver, problem: Problem) -> str:
    """Why ``solver`` does not support ``problem``, in a few words for a table cell.

    The first of the boundary condition, the dimension and the domain that
    the solver does not declare; then, for a solver on :class:`Monotone17`,
    the stencils of the problem's dimension when the scheme's stencil is of
    another (``3D: stencil=19`` for ``FDMonotone``'s default 17-point stencil
    on a 3D problem); for a solver on :class:`QuadraticForm`, the reason
    :meth:`QuadraticForm.refusal` gives for its reference point (``f(x0) <=
    0`` for ``DirichletQSPINN`` on T20); ``unsupported`` if none of these is
    the reason (unequal sides of a box, say).
    """
    if not isinstance(problem.boundary, solver.boundaries):
        return "/".join(kind.__name__ for kind in solver.boundaries) + " only"
    if solver.dims is not None and problem.dim not in solver.dims:
        return "/".join(f"{d}D" for d in solver.dims) + " only"
    if not isinstance(problem.domain, solver.domains):
        return f"not on a {type(problem.domain).__name__}"
    scheme = getattr(solver, "scheme", None)
    if isinstance(scheme, Monotone17) and scheme.dim != problem.dim:
        stencils = get_args(get_type_hints(Monotone17)["stencil"])
        return f"{problem.dim}D: stencil=" + "/".join(
            str(s) for s in stencils if stencil_dimension(s) == problem.dim
        )
    network = getattr(solver, "network", None)
    if isinstance(network, QuadraticForm):
        return network.refusal(problem) or "unsupported"
    return "unsupported"


def threshold(problem: Problem) -> float:
    """The success threshold of a problem: :data:`RELATIVE_THRESHOLD` of the oscillation."""
    if problem.exact is None:
        raise ValueError(f"{problem.name} has no exact solution")
    u = problem.exact(shared_grid(problem.domain))
    return RELATIVE_THRESHOLD * float(np.max(u) - np.min(u))


def read(path: Path = CSV) -> list[dict[str, str]]:
    """The rows of comparison.csv, as strings."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def order(row: Mapping[str, Any]) -> tuple[int, int, int]:
    """Sort key of a row: problem, preset, seed, in the order of the table."""
    seed = row["seed"]
    return (
        SHARED_TEST_SET.index(row["problem"]),
        list(SOLVERS).index(row["solver"]),
        -1 if seed in ("", None) else int(seed),
    )


def merge(
    old: Iterable[Mapping[str, Any]], new: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The rows of ``old`` whose pairs ``new`` does not cover, with ``new``, sorted.

    A rerun of some pairs replaces every row of those pairs, whatever their
    seeds; a missing field is empty.
    """
    new = list(new)
    pairs = {(r["solver"], r["problem"]) for r in new}
    kept = [r for r in old if (r["solver"], r["problem"]) not in pairs]
    full = [{key: row.get(key, "") for key in FIELDS} for row in [*kept, *new]]
    return sorted(full, key=order)


def number(text: str) -> float:
    """A CSV number; an empty field is NaN."""
    return float(text) if text else math.nan


def scientific(x: float) -> str:
    """Three significant digits, exponent without padding: ``7.06e-3``."""
    if math.isinf(x):
        return "inf"
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def percent(x: float) -> str:
    """A fraction as a percentage with three significant digits, 0 as ``0``."""
    return "0" if x == 0 else f"{100 * x:.3g}%"


def seconds(x: float) -> str:
    """Seconds, whole from 100 on, else three significant digits."""
    return f"{x:.0f}" if x >= 100 else f"{x:.3g}"


def _metric(value: str, form) -> str:
    """A formatted metric, ``n/a`` for NaN (a non-finite Hessian)."""
    x = number(value)
    return "n/a" if math.isnan(x) else form(x)


def _classical_cell(row: Mapping[str, str], key: str) -> str:
    """One run: the error with its grid, ``\\*`` if the iteration did not converge."""
    if row["status"] == "failed":
        return f"failed ({row['n']})" if key == "max" else "failed"
    mark = "" if row["converged"] == "True" else " \\*"
    if key == "max":
        return f"{scientific(number(row['max']))} ({row['n']}){mark}"
    form = percent if key == "nonconvex_fraction" else seconds
    return _metric(row[key], form) + mark


def _neural_cell(rows: Sequence[Mapping[str, str]], key: str, problem: Problem) -> str:
    """Seeds: the error's median, 5% to 95% band and successes; else the median."""
    errors = [None if r["status"] == "failed" else number(r["max"]) for r in rows]
    summary = summarize_errors(errors, threshold(problem))
    if math.isinf(summary.median):
        return f"failed ({summary.failed}/{summary.seeds})"
    if key == "max":
        band = f"[{scientific(summary.q05)}, {scientific(summary.q95)}]"
        successes = round(summary.success_rate * summary.seeds)
        return f"{scientific(summary.median)} {band}, {successes}/{summary.seeds}"
    values = [number(r[key]) for r in rows if r["status"] != "failed"]
    form = percent if key == "nonconvex_fraction" else seconds
    return "n/a" if np.isnan(values).any() else form(float(np.median(values)))


def cell(rows: Sequence[Mapping[str, str]], key: str) -> str:
    """The table cell of one preset on one problem for ``key``."""
    if rows[0]["status"] == "unsupported":
        return rows[0]["reason"]
    if rows[0]["seed"]:
        return _neural_cell(rows, key, CATALOGUE[rows[0]["problem"]])
    return _classical_cell(rows[0], key)


def group(
    rows: Iterable[Mapping[str, str]],
) -> dict[tuple[str, str], list[Mapping[str, str]]]:
    """The rows by ``(problem, solver)``."""
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["problem"], row["solver"]), []).append(row)
    return groups


TABLES = (
    ("max", "Max error at the primary points"),
    ("nonconvex_fraction", "Non-convex fraction"),
    ("time", "Time in seconds"),
)


def render(rows: Iterable[Mapping[str, str]]) -> str:
    """The README block between the markers: the three tables and their notes."""
    groups = group(rows)
    missing = [
        f"{solver} on {problem}"
        for problem in SHARED_TEST_SET
        for solver in SOLVERS
        if (problem, solver) not in groups
    ]
    if missing:
        raise ValueError(f"comparison.csv lacks {', '.join(missing)}")
    lines = []
    for key, title in TABLES:
        lines += [
            f"**{title}**",
            "",
            "| Problem | " + " | ".join(f"`{name}`" for name in SOLVERS) + " |",
            "|---" * (len(SOLVERS) + 1) + "|",
        ]
        for problem in SHARED_TEST_SET:
            cells = [cell(groups[problem, solver], key) for solver in SOLVERS]
            lines.append(f"| {problem} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines + notes(groups))


def _listing(items: Iterable[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    words = list(items)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _one(rows: Iterable[Mapping[str, str]], key: str) -> str:
    """The value of ``key`` shared by ``rows``; ValueError if they differ."""
    values = {row[key] for row in rows}
    if len(values) != 1:
        raise ValueError(f"comparison.csv mixes {key}: {', '.join(sorted(values))}")
    return values.pop()


def notes(groups: Mapping[tuple[str, str], Sequence[Mapping[str, str]]]) -> list[str]:
    """The notes under the tables, each derived from the rows it describes."""
    runs = [r for rows in groups.values() for r in rows if r["status"] != "unsupported"]
    threads, device, cpu = (_one(runs, key) for key in ("threads", "device", "cpu"))
    classical = [r for r in runs if not r["seed"]]
    seeds = sorted({int(r["seed"]) for r in runs if r["seed"]})
    steps: dict[str, list[str]] = {}
    for problem in SHARED_TEST_SET:
        if any(r["problem"] == problem for r in classical):
            box = CATALOGUE[problem].domain.bbox
            steps.setdefault(f"{box.hi[0] - box.lo[0]:g}/n", []).append(problem)
    reasons: dict[str, list[str]] = {}
    for (problem, _), rows in sorted(groups.items(), key=lambda g: order(g[1][0])):
        if rows[0]["status"] == "unsupported":
            problems = reasons.setdefault(rows[0]["reason"], [])
            if problem not in problems:
                problems.append(problem)
    planned = [
        f"{PLANNED[reason]} ({_listing(problems)})"
        for reason, problems in reasons.items()
        if reason in PLANNED
    ]
    return [
        "A classical cell is one run on the grid `n` in parentheses, the largest "
        "`N` of the papers, with "
        + ", ".join(f"`h = {h}` on {_listing(p)}" for h, p in steps.items())
        + " (ADR 0001), and its error at its own nodes; "
        "`\\*` marks a run whose iteration stopped before its tolerance.",
        f"A neural cell is over seeds {seeds[0]} to {seeds[-1]}: the median, the 5% "
        "to 95% band and the number of seeds whose max error on the shared "
        f"evaluation grid is below {RELATIVE_THRESHOLD:.0%} of the exact solution's "
        "oscillation `max u - min u` there; a failed seed counts as an infinite "
        "error (ADR 0007).",
        "The non-convex fraction and the time of a neural cell are medians over the "
        "seeds that did not fail.",
        f"Every run took {threads} CPU thread{'' if threads == '1' else 's'} and "
        f"ran on the device `{device}`; CPU: {cpu}.",
        "A cell with a reason marks a preset that does not support the problem: "
        + "; ".join(f"`{r}` on {_listing(p)}" for r, p in reasons.items())
        + ".",
    ] + (
        [f"The roadmap's 0.2 gives the finite-difference presets {_listing(planned)}."]
        if planned
        else []
    )


def replace_block(text: str, block: str) -> str:
    """``text`` with the lines between the markers replaced by ``block``."""
    start, end = text.find(START), text.find(END)
    if start < 0 or end < start:
        raise ValueError("README.md lacks the comparison table markers")
    return text[: start + len(START)] + "\n" + block + "\n" + text[end:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if README.md is not current"
    )
    args = parser.parse_args(argv)
    text = README.read_text()
    new = replace_block(text, render(read()))
    if args.check:
        if new != text:
            print("README.md is not current: run python benchmarks/comparison.py")
            return 1
        return 0
    README.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
