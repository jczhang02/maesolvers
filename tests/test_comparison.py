"""The comparison table: benchmarks/comparison.py, its cells and its README block."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from maesolvers import CATALOGUE, SOLVERS, T1, T8, T9, T12, T20, NeuralSolver
from maesolvers.catalogue import SHARED_TEST_SET

ROOT = Path(__file__).parent.parent


def _load():
    """benchmarks/comparison.py as a module; it needs numpy and maesolvers only."""
    spec = importlib.util.spec_from_file_location(
        "comparison", ROOT / "benchmarks" / "comparison.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = _load()


def _row(**fields: object) -> dict[str, str]:
    """A CSV row as comparison.read returns it: every field a string."""
    row = dict.fromkeys(comparison.FIELDS, "")
    row.update({key: str(value) for key, value in fields.items()})
    return row


def _classical(**fields: object) -> dict[str, str]:
    defaults = {
        "solver": "FDNewton",
        "problem": "T1",
        "n": 361,
        "status": "ok",
        "max": 5.276e-7,
        "nonconvex_fraction": 0.0,
        "time": 12.3,
        "converged": True,
    }
    return _row(**{**defaults, **fields})


def _seeds(errors: list[float | None], problem: str = "T1") -> list[dict[str, str]]:
    """Neural rows, one per seed; None for a failed seed."""
    return [
        _row(
            solver="PenaltyMLP",
            problem=problem,
            seed=seed,
            status="failed" if error is None else "ok",
            max="" if error is None else error,
            nonconvex_fraction=0.01 * seed,
            time=400.0 + seed,
        )
        for seed, error in enumerate(errors)
    ]


# Formatting


def test_numbers_print_with_three_significant_digits():
    assert comparison.scientific(7.0612e-3) == "7.06e-3"
    assert comparison.scientific(12.5) == "1.25e1"
    assert comparison.scientific(0.0) == "0"
    assert comparison.scientific(math.inf) == "inf"
    assert comparison.percent(0.0) == "0"
    assert comparison.percent(0.06693) == "6.69%"
    assert comparison.seconds(911.02) == "911"
    assert comparison.seconds(0.05124) == "0.0512"


# Support


@pytest.mark.parametrize(
    ("solver", "problem", "reason"),
    [
        ("FDNewton", T12, "unsupported until 0.2"),
        ("DirichletICNN", T12, "unsupported until 0.2"),
        ("FDNewton", T9, "2D only"),
        ("PenaltyMLP", T9, "2D only"),
        ("FDMonotone", T8, "not on a Disk"),
    ],
)
def test_unsupported_reason_says_why(solver, problem, reason):
    instance = SOLVERS[solver]()
    assert not instance.supports(problem)
    assert comparison.unsupported_reason(instance, problem) == reason


def test_threshold_is_a_share_of_the_largest_exact_value():
    # max |u| of exp(|x|^2 / 2) on the unit square is e, at (1, 1)
    assert comparison.threshold(T1) == pytest.approx(1e-2 * math.e)
    # ((|x - x0| - 0.2)^+)^2 / 2 is largest at the corners
    assert comparison.threshold(T20) == pytest.approx(1e-2 * (0.5**0.5 - 0.2) ** 2 / 2)


# Cells


def test_classical_cell_gives_the_error_and_its_grid():
    row = _classical()
    assert comparison.cell([row], "max") == "5.28e-7 (361)"
    assert comparison.cell([row], "nonconvex_fraction") == "0"
    assert comparison.cell([row], "time") == "12.3"


def test_classical_cell_marks_an_unconverged_run():
    row = _classical(converged=False)
    assert comparison.cell([row], "max") == "5.28e-7 (361) \\*"
    assert comparison.cell([row], "time") == "12.3 \\*"


def test_classical_cell_of_a_failed_run():
    row = _classical(status="failed", max="", reason="LinAlgError: singular")
    assert comparison.cell([row], "max") == "failed (361)"
    assert comparison.cell([row], "time") == "failed"


def test_classical_cell_without_a_hessian_value():
    row = _classical(nonconvex_fraction=math.nan)
    assert comparison.cell([row], "nonconvex_fraction") == "n/a"


def test_neural_cell_gives_the_band_and_the_successes():
    # threshold on T1: 1e-2 e = 2.72e-2, so seeds 0 to 4 (up to 2.5e-2) succeed
    errors = [1e-3 * (1 + seed) ** 2 for seed in range(10)]
    rows = _seeds(errors)
    expected = np.quantile(errors, [0.5, 0.05, 0.95])
    median, q05, q95 = (comparison.scientific(x) for x in expected)
    assert comparison.cell(rows, "max") == f"{median} [{q05}, {q95}], 5/10"
    assert comparison.cell(rows, "nonconvex_fraction") == "4.5%"
    assert comparison.cell(rows, "time") == "404"


def test_neural_cell_counts_a_failed_seed_as_infinite():
    rows = _seeds([1e-3] * 9 + [None])
    assert comparison.cell(rows, "max") == "1.00e-3 [1.00e-3, inf], 9/10"
    # the medians of the other metrics leave the failed seed out
    assert comparison.cell(rows, "time") == "404"


def test_neural_cell_fails_when_half_the_seeds_fail():
    rows = _seeds([1e-3] * 5 + [None] * 5)
    for key in ("max", "nonconvex_fraction", "time"):
        assert comparison.cell(rows, key) == "failed (5/10)"


def test_unsupported_cell_gives_the_reason():
    row = _row(solver="FDNewton", problem="T12", status="unsupported", reason="r")
    assert comparison.cell([row], "max") == "r"


# The block


def test_render_needs_every_pair():
    with pytest.raises(ValueError, match="lacks FDNewton on T1"):
        comparison.render([])


def test_replace_block_keeps_the_text_around_the_markers():
    text = f"intro\n{comparison.START}\nold\n{comparison.END}\nrest\n"
    new = comparison.replace_block(text, "new")
    assert new == f"intro\n{comparison.START}\nnew\n{comparison.END}\nrest\n"
    with pytest.raises(ValueError, match="markers"):
        comparison.replace_block("no markers", "new")


# The receipt


def test_readme_table_is_current():
    """README's block is what comparison.py writes from comparison.csv."""
    readme = comparison.README.read_text()
    block = comparison.render(comparison.read())
    assert comparison.replace_block(readme, block) == readme, (
        "README.md is not current: run `just comparison-table`"
    )


def test_csv_covers_every_preset_on_the_shared_test_set():
    """A new preset, or a change of support, needs a rerun of run_comparison.py."""
    groups = comparison.group(comparison.read())
    assert {solver for _, solver in groups} == set(SOLVERS)
    assert {problem for problem, _ in groups} == set(SHARED_TEST_SET)
    for (problem, name), rows in groups.items():
        solver = SOLVERS[name]()
        if not solver.supports(CATALOGUE[problem]):
            assert [r["status"] for r in rows] == ["unsupported"], (problem, name)
        elif isinstance(solver, NeuralSolver):
            assert [r["seed"] for r in rows] == [str(s) for s in range(10)]
        else:
            assert len(rows) == 1 and rows[0]["n"] == "361", (problem, name)
        assert all(r["status"] in comparison.STATUSES for r in rows)
