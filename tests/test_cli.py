"""The ``maesolvers solve`` command line (ADR 0005)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from maesolvers import T1, FDNewton, Newton, Standard9, from_config
from maesolvers.metrics import evaluate

pytest.importorskip("jsonargparse")

from maesolvers.cli import main  # noqa: E402

COMPOSED_YAML = textwrap.dedent(
    """\
    solver:
      class_path: Newton
      init_args:
        scheme:
          class_path: Standard9
          init_args:
            n: 16
    problem: T1
    """
)


def _solve(tmp_path, *args):
    """Run ``maesolvers solve ARGS --output FILE`` and return the written record."""
    output = tmp_path / "out.json"
    main(["solve", *args, "--output", str(output)])
    return json.loads(output.read_text())


def test_solve_prints_evaluate_as_a_table(capsys):
    main(["solve", "--solver", "FDNewton", "--solver.n", "8", "--problem", "T1"])
    lines = capsys.readouterr().out.splitlines()
    metrics = evaluate(FDNewton(n=8).solve(T1))
    assert lines[0] == "FDNewton on T1: converged after 3 iterations"
    rows = dict(line.split() for line in lines[1:])
    assert list(rows) == list(metrics)
    assert float(rows["max"]) == pytest.approx(metrics["max"], rel=1e-5)
    assert int(rows["unknowns"]) == metrics["unknowns"]


def test_solve_writes_the_rerun_config_metrics_and_info(tmp_path):
    record = _solve(
        tmp_path, "--solver", "FDNewton", "--solver.n", "8", "--problem", "T1"
    )
    result = FDNewton(n=8).solve(T1)
    assert record["config"] == {"solver": result.hparams, "problem": "T1"}
    assert from_config(record["config"]) == (FDNewton(n=8), T1)
    assert record["metrics"].keys() == evaluate(result).keys()
    assert record["metrics"]["max"] == pytest.approx(evaluate(result)["max"])
    assert record["info"]["converged"] is True
    assert record["info"]["history"] == pytest.approx(list(result.info["history"]))


def test_keywords_dict_yaml_and_command_line_build_the_same_solver(tmp_path):
    expected = Newton(scheme=Standard9(n=8), tol=1e-9)
    as_dict = {
        "solver": {
            "class_path": "Newton",
            "init_args": {
                "tol": 1e-9,
                "scheme": {"class_path": "Standard9", "init_args": {"n": 8}},
            },
        },
        "problem": "T1",
    }
    path = tmp_path / "run.yaml"
    path.write_text(COMPOSED_YAML.replace("n: 16", "n: 8") + "")
    from_yaml, _ = from_config(path)
    args = ["--solver", "Newton", "--solver.tol", "1e-9"]
    args += [
        "--solver.scheme",
        "Standard9",
        "--solver.scheme.n",
        "8",
        "--problem",
        "T1",
    ]
    from_cli, _ = from_config(_solve(tmp_path, *args)["config"])
    assert from_config(as_dict)[0] == expected
    assert from_yaml == Newton(scheme=Standard9(n=8))
    assert from_cli == expected


def test_an_override_after_the_config_file_wins(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(COMPOSED_YAML)
    record = _solve(tmp_path, "--config", str(path), "--solver.scheme.n", "8")
    assert from_config(record["config"]) == (Newton(scheme=Standard9(n=8)), T1)


def test_the_problem_is_set_on_the_command_line(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(COMPOSED_YAML)
    record = _solve(tmp_path, "--config", str(path), "--problem", "T20")
    assert record["config"]["problem"] == "T20"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--solver", "FDNewton", "--solver.nn", "8", "--problem", "T1"], "'nn'"),
        (
            ["--solver", "FDNewton", "--solver.n", "1", "--problem", "T1"],
            "n must be at least 2",
        ),
        (["--solver", "FDNewton", "--problem", "T99"], "T99"),
        (["--solver", "FDNewton"], "problem"),
        (["--solver", "FDNewton", "--problem", "T12"], "FDNewton does not support T12"),
    ],
    ids=[
        "misspelt key",
        "rejected value",
        "unknown problem",
        "no problem",
        "unsupported",
    ],
)
def test_a_bad_argument_exits_with_status_2_and_names_it(capsys, args, message):
    with pytest.raises(SystemExit) as exit_info:
        main(["solve", *args])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_python_dash_m_runs_the_command():
    args = ["solve", "--solver", "FDNewton", "--solver.n", "8", "--problem", "T1"]
    out = subprocess.run(
        [sys.executable, "-m", "maesolvers", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("FDNewton on T1: converged")
