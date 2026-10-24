"""The comparison table: benchmarks/comparison.py, its cells and its README block."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from maesolvers import (
    CATALOGUE,
    SOLVERS,
    T1,
    T8,
    T9,
    T10,
    T12,
    T20,
    Box,
    NeuralSolver,
    PenaltyMLP,
    Problem,
)
from maesolvers.catalogue import SHARED_TEST_SET
from maesolvers.neural import BFGS

ROOT = Path(__file__).parent.parent


def _load(name: str):
    """benchmarks/<name>.py as a module, its ``import comparison`` bound to ours.

    Both scripts import numpy and maesolvers only at module level; the runner
    imports torch inside a neural seed's run and mlflow only to log one.
    """
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmarks" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"comparison": globals().get("comparison")}):
        spec.loader.exec_module(module)
    return module


comparison = _load("comparison")
run_comparison = _load("run_comparison")


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


RECTANGLE = Problem("RECT", Box((0.0, 0.0), (1.0, 2.0)), T1.f, T1.boundary)
"""A 2D Dirichlet problem on a box with unequal sides."""


@pytest.mark.parametrize(
    ("solver", "problem", "reason"),
    [
        ("FDNewton", T12, "Dirichlet only"),
        ("FDMonotone", T12, "Dirichlet only"),
        ("DirichletICNN", T12, "Dirichlet only"),
        ("FDNewton", T9, "2D only"),
        ("PenaltyMLP", T9, "2D only"),
        ("FDMonotone", T9, "3D: stencil=19"),
        ("FDMonotone", T10, "2D/3D only"),
        ("FDMonotone", T8, "not on a Disk"),
        ("FDMonotone", RECTANGLE, "unsupported"),
        ("DirichletQSPINN", T20, "f(x0) <= 0"),
        ("TransportQSPINN", T8, "Transport only"),
    ],
)
def test_unsupported_reason_says_why(solver, problem, reason):
    instance = SOLVERS[solver]()
    assert not instance.supports(problem)
    assert comparison.unsupported_reason(instance, problem) == reason


@pytest.mark.parametrize(
    ("reference", "problem", "reason"),
    [
        ((5.0, 5.0), T1, "x0 outside"),
        ((0.5, 0.5, 0.5), T1, "reference dim"),
        ((0.5, 0.5), T20, "f(x0) <= 0"),
    ],
)
def test_unsupported_reason_names_the_reference_point(reference, problem, reason):
    solver = SOLVERS["DirichletQSPINN"](reference=reference)
    assert not solver.supports(problem)
    assert comparison.unsupported_reason(solver, problem) == reason


def test_preset_options_go_to_the_presets_that_take_them():
    options = {"n": 361, "max_steps": 10, "seed": 3, "accelerator": None}
    assert comparison.preset_options("FDNewton", options) == {"n": 361}
    assert comparison.preset_options("PenaltyMLP", options) == {
        "max_steps": 10,
        "seed": 3,
    }


def test_with_parts_swaps_the_named_part_and_keeps_every_other_field():
    preset = PenaltyMLP(seed=3, max_steps=7, tol=1e-6)
    solver = comparison.with_parts(preset, optimizer=BFGS())
    assert type(solver) is NeuralSolver
    assert solver.optimizer == BFGS()
    for field in dataclasses.fields(NeuralSolver):
        if field.name != "optimizer":
            assert getattr(solver, field.name) is getattr(preset, field.name)


def test_with_parts_refuses_a_name_that_is_not_a_field():
    with pytest.raises(TypeError, match="not fields of NeuralSolver: optimiser"):
        comparison.with_parts(PenaltyMLP(), optimiser=BFGS())


def test_threshold_is_a_share_of_the_exact_oscillation():
    # exp(|x|^2 / 2) on the unit square runs from 1 at 0 to e at (1, 1)
    assert comparison.threshold(T1) == pytest.approx(1e-2 * (math.e - 1))
    # ((|x - x0| - 0.2)^+)^2 / 2 is 0 on the disk, largest at the corners
    assert comparison.threshold(T20) == pytest.approx(1e-2 * (0.5**0.5 - 0.2) ** 2 / 2)


def test_threshold_ignores_a_constant_added_to_the_solution():
    shifted = Problem(
        "T1+1", T1.domain, T1.f, T1.boundary, exact=lambda x: T1.exact(x) + 1
    )
    assert comparison.threshold(shifted) == pytest.approx(comparison.threshold(T1))


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
    # threshold on T1: 1e-2 (e - 1) = 1.72e-2, so seeds 0 to 3 (up to 1.6e-2) succeed
    errors = [1e-3 * (1 + seed) ** 2 for seed in range(10)]
    rows = _seeds(errors)
    expected = np.quantile(errors, [0.5, 0.05, 0.95])
    median, q05, q95 = (comparison.scientific(x) for x in expected)
    assert comparison.cell(rows, "max") == f"{median} [{q05}, {q95}], 4/10"
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


def _machine(row: dict[str, str]) -> dict[str, str]:
    return {**row, "threads": "1", "cpu": "Some CPU", "device": "cpu"}


def test_notes_derive_from_the_rows():
    rows = [
        _classical(),
        _classical(problem="T8"),
        *_seeds([1e-3, 2e-3]),
        _row(solver="FDNewton", problem="T9", status="unsupported", reason="2D only"),
        _row(solver="FDNewton", problem="T10", status="unsupported", reason="2D only"),
        _row(solver="FDMonotone", problem="T8", status="unsupported", reason="r"),
        _row(
            solver="FDNewton",
            problem="T12",
            status="unsupported",
            reason="Dirichlet only",
        ),
    ]
    notes = comparison.notes(comparison.group(map(_machine, rows)))
    assert "`h = 1/n` on T1, `h = 2/n` on T8 (ADR 0001)" in notes[0]
    assert "over seeds 0 to 1:" in notes[1]
    assert notes[3] == (
        "Every run took 1 CPU thread and ran on the device `cpu`; CPU: Some CPU."
    )
    assert notes[4].endswith(
        ": `r` on T8; `Dirichlet only` on T12; `2D only` on T9 and T10."
    )
    assert len(notes) == 5


def test_notes_need_one_machine():
    rows = [_machine(_classical()), {**_machine(_classical(problem="T2")), "cpu": "B"}]
    with pytest.raises(ValueError, match="mixes cpu"):
        comparison.notes(comparison.group(rows))


def test_merge_replaces_the_rerun_pairs_only():
    old = [
        _classical(problem="T2", max=1.0),
        *_seeds([1e-3, 2e-3, 3e-3]),
        _classical(problem="T1", max=2.0),
    ]
    new = _seeds([4e-3, 5e-3])  # a rerun of PenaltyMLP on T1 with two seeds
    merged = comparison.merge(old, new)
    assert [(r["problem"], r["solver"], r["seed"]) for r in merged] == [
        ("T1", "FDNewton", ""),
        ("T1", "PenaltyMLP", "0"),
        ("T1", "PenaltyMLP", "1"),
        ("T2", "FDNewton", ""),
    ]
    assert [r["max"] for r in merged] == ["2.0", "0.004", "0.005", "1.0"]
    assert all(set(r) == set(comparison.FIELDS) for r in merged)


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
    """A new preset, a change of support, or of a preset's defaults, needs a rerun."""
    groups = comparison.group(comparison.read())
    assert {solver for _, solver in groups} == set(SOLVERS)
    assert {problem for problem, _ in groups} == set(SHARED_TEST_SET)
    for (problem_id, name), rows in groups.items():
        problem, solver = CATALOGUE[problem_id], SOLVERS[name]()
        assert all(r["status"] in comparison.STATUSES for r in rows)
        if not solver.supports(problem):
            reason = comparison.unsupported_reason(solver, problem)
            assert [(r["status"], r["reason"]) for r in rows] == [
                ("unsupported", reason)
            ], (problem_id, name)
            continue
        if isinstance(solver, NeuralSolver):
            assert [r["seed"] for r in rows] == [str(s) for s in comparison.SEEDS]
        else:
            assert len(rows) == 1 and rows[0]["n"] == str(comparison.N), (problem, name)
        for row in rows:
            seed = int(row["seed"]) if row["seed"] else None
            options = comparison.preset_options(name, {"n": comparison.N, "seed": seed})
            hparams = SOLVERS[name](**options).hparams
            assert json.loads(row["hparams"]) == hparams, (problem_id, name, seed)


# The runner


def test_tasks_give_a_run_per_seed_neural_first_and_the_unsupported_rows():
    options = {"n": 8, "max_steps": None}
    runs, unsupported = run_comparison.tasks(
        ["FDNewton", "PenaltyMLP", "DirichletICNN"],
        ["T1", "T9"],
        [0, 1],
        options,
        False,
    )
    assert [run[:3] for run in runs] == [
        ("DirichletICNN", "T9", 0),
        ("DirichletICNN", "T9", 1),
        ("PenaltyMLP", "T1", 0),
        ("PenaltyMLP", "T1", 1),
        ("DirichletICNN", "T1", 0),
        ("DirichletICNN", "T1", 1),
        ("FDNewton", "T1", None),
    ]
    assert all(run[3:] == (options, False) for run in runs)
    assert unsupported == [
        {"solver": name, "problem": "T9", "status": "unsupported", "reason": "2D only"}
        for name in ("FDNewton", "PenaltyMLP")
    ]


def test_run_one_gives_a_full_row_for_a_classical_run():
    row = run_comparison.run_one(("FDNewton", "T1", None, {"n": 8, "seed": 0}, False))
    assert set(row) == set(comparison.FIELDS)
    assert row["status"] == "ok" and row["reason"] == ""
    assert (row["solver"], row["problem"], row["seed"], row["n"]) == (
        "FDNewton",
        "T1",
        "",
        8,
    )
    assert json.loads(row["hparams"]) == SOLVERS["FDNewton"](n=8).hparams
    assert row["converged"] is True and row["stop"] == "" and row["device"] == "cpu"
    assert 0.0 < row["max"] < 1e-2 and row["threads"] == 1


@dataclasses.dataclass(frozen=True)
class _Raises(SOLVERS["FDNewton"]):
    def setup(self, problem):
        raise RuntimeError("no  grid\nhere")


@dataclasses.dataclass(frozen=True)
class _Diverges(SOLVERS["FDNewton"]):
    def finalize(self, problem, context, state):
        solution = super().finalize(problem, context, state)
        return dataclasses.replace(solution, info={**solution.info, "stop": "diverged"})


@dataclasses.dataclass(frozen=True)
class _NotFinite(SOLVERS["FDNewton"]):
    def finalize(self, problem, context, state):
        solution = super().finalize(problem, context, state)
        nan = np.full_like(solution.values, np.nan)
        return dataclasses.replace(solution, values=nan)


@pytest.mark.parametrize(
    ("solver", "reason"),
    [
        (_Raises, "RuntimeError: no grid here"),
        (_Diverges, "stopped as diverged"),
        (_NotFinite, "non-finite"),
    ],
)
def test_run_one_reports_a_failed_run_as_a_row(monkeypatch, solver, reason):
    solvers = {**SOLVERS, "Broken": solver}  # SOLVERS itself is read-only
    for module in (comparison, run_comparison):
        monkeypatch.setattr(module, "SOLVERS", solvers)
    row = run_comparison.run_one(("Broken", "T1", None, {"n": 8}, False))
    assert (row["status"], row["reason"]) == ("failed", reason)
    assert row["solver"] == "Broken"
    assert json.loads(row["hparams"]) == solver(n=8).hparams
