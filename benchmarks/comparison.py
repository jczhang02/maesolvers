"""Write the README comparison table from comparison.csv, or check that it is current.

comparison.csv holds one row per run of a preset of ``SOLVERS`` on a problem of
the shared test set (``run_comparison.py`` writes it on a100server): one row
for a classical preset, one per seed for a neural one, and one row with status
``unsupported`` for a pair the preset does not support. This script turns the
rows into three tables, the max error, the non-convex fraction and the time,
with the problems as rows and the presets as columns, and writes them into
README.md between the two marker lines. It needs numpy and maesolvers only, so
the test that README is current (tests/test_comparison.py) runs without
extras. From the repository root::

    python benchmarks/comparison.py          # rewrite the README block
    python benchmarks/comparison.py --check  # exit 1 if README is not current
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from maesolvers import CATALOGUE, SOLVERS, Problem, Solver, Transport
from maesolvers.catalogue import SHARED_TEST_SET
from maesolvers.metrics import shared_grid, summarize_errors

HERE = Path(__file__).parent
CSV = HERE / "comparison.csv"
README = HERE.parent / "README.md"

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
    "commit",
)
"""The columns of comparison.csv."""

STATUSES = ("ok", "failed", "unsupported")
"""``ok``: a finite error; ``failed``: an exception, a non-finite error, or a
neural run that stopped as ``"diverged"`` (ADR 0007); ``unsupported``: the
preset does not support the problem, and ``reason`` says why."""

RELATIVE_THRESHOLD = 1e-2
"""A neural seed succeeds when its max error is below this share of the exact
solution's largest absolute value on the shared evaluation grid."""


def unsupported_reason(solver: Solver, problem: Problem) -> str:
    """Why ``solver`` does not support ``problem``, in a few words for a table cell."""
    if isinstance(problem.boundary, Transport):
        return "unsupported until 0.2"
    dims = type(solver).dims
    if dims is not None and problem.domain.dim not in dims:
        return "/".join(f"{d}D" for d in dims) + " only"
    return f"not on a {type(problem.domain).__name__}"


def threshold(problem: Problem) -> float:
    """The success threshold of a problem: :data:`RELATIVE_THRESHOLD` of max ``|u|``."""
    if problem.exact is None:
        raise ValueError(f"{problem.name} has no exact solution")
    u = problem.exact(shared_grid(problem.domain))
    return RELATIVE_THRESHOLD * float(np.max(np.abs(u)))


def read(path: Path = CSV) -> list[dict[str, str]]:
    """The rows of comparison.csv, as strings."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


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
    seeds = max(len(g) for g in groups.values())
    lines += [
        "A classical cell is one run on the grid `n` in parentheses, the largest "
        "`N` of the papers (`h = 1/n`, ADR 0001), at its own nodes; "
        "`\\*` marks a run whose iteration stopped before its tolerance.",
        f"A neural cell is over seeds 0 to {seeds - 1}: the median, the 5% to 95% "
        "band and the number of seeds whose max error on the shared evaluation grid "
        f"is below {RELATIVE_THRESHOLD:.0%} of the exact solution's largest "
        "absolute value there; the non-convex fraction and the time are medians, "
        "and a failed seed counts as an infinite error (ADR 0007).",
        "Every run used one CPU thread on a100server; the neural presets train on "
        "the CPU, their default.",
        "A cell with a reason marks a preset that does not support the problem: "
        "`unsupported until 0.2` for the transport boundary condition of T12, `2D only` for a "
        "2D method on T9 (3D) and T10 (4D), `not on a Disk` for a box-only scheme.",
    ]
    return "\n".join(lines)


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
