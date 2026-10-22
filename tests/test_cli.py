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

import yaml  # a jsonargparse dependency

from maesolvers.cli import main

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
    assert record["run"] == {"solver": result.hparams, "problem": "T1"}
    assert from_config(record["run"]["solver"]) == FDNewton(n=8)
    assert record["metrics"].keys() == evaluate(result).keys()
    assert record["metrics"]["max"] == pytest.approx(evaluate(result)["max"])
    assert record["info"]["converged"] is True
    assert record["info"]["history"] == pytest.approx(list(result.info["history"]))


def test_keywords_dict_yaml_and_command_line_build_the_same_solver(tmp_path):
    expected = Newton(scheme=Standard9(n=8), tol=1e-9)
    as_dict = {
        "class_path": "Newton",
        "init_args": {
            "tol": 1e-9,
            "scheme": {"class_path": "Standard9", "init_args": {"n": 8}},
        },
    }
    path = tmp_path / "solver.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            class_path: Newton
            init_args:
              tol: 1e-9
              scheme:
                class_path: Standard9
                init_args:
                  n: 8
            """
        )
    )
    args = ["--solver", "Newton", "--solver.tol", "1e-9"]
    args += ["--solver.scheme", "Standard9", "--solver.scheme.n", "8"]
    from_cli = _solve(tmp_path, *args, "--problem", "T1")["run"]["solver"]
    assert from_config(as_dict) == expected
    assert from_config(path) == expected
    assert from_config(from_cli) == expected


def test_an_override_after_the_config_file_wins(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(COMPOSED_YAML)
    record = _solve(tmp_path, "--config", str(path), "--solver.scheme.n", "8")
    assert from_config(record["run"]["solver"]) == Newton(scheme=Standard9(n=8))
    assert record["run"]["problem"] == "T1"


def test_the_problem_is_set_on_the_command_line(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(COMPOSED_YAML)
    record = _solve(tmp_path, "--config", str(path), "--problem", "T20")
    assert record["run"]["problem"] == "T20"


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
        (
            ["--solver", "FDNewton", "--problem", "T1", "--output", "/no/dir/out.json"],
            "output",
        ),
    ],
    ids=[
        "misspelt key",
        "rejected value",
        "unknown problem",
        "no problem",
        "unsupported",
        "output dir missing",
    ],
)
def test_a_bad_argument_exits_with_status_2_and_names_it(capsys, args, message):
    with pytest.raises(SystemExit) as exit_info:
        main(["solve", *args])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_the_output_record_is_not_a_config_file(tmp_path, capsys):
    record = tmp_path / "out.json"
    _solve(tmp_path, "--solver", "FDNewton", "--solver.n", "8", "--problem", "T1")
    with pytest.raises(SystemExit) as exit_info:
        main(["solve", "--config", str(record)])
    assert exit_info.value.code == 2
    assert "run.solver" in capsys.readouterr().err


def test_an_output_path_in_a_config_file_is_relative_to_the_file(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "run.yaml"
    path.write_text(COMPOSED_YAML + "output: out.json\n")
    monkeypatch.chdir(tmp_path)
    main(["solve", "--config", str(path)])
    assert (tmp_path / "configs" / "out.json").is_file()
    assert not (tmp_path / "out.json").exists()


def test_a_printed_config_reruns(tmp_path, capsys):
    path = tmp_path / "run.yaml"
    path.write_text(COMPOSED_YAML)
    with pytest.raises(SystemExit):
        main(["solve", "--config", str(path), "--solver.tol", "1e-9", "--print_config"])
    printed = tmp_path / "printed.yaml"
    printed.write_text(capsys.readouterr().out)
    record = _solve(tmp_path, "--config", str(printed))
    assert from_config(record["run"]["solver"]) == Newton(
        scheme=Standard9(n=16), tol=1e-9
    )


@pytest.mark.parametrize(
    "solver",
    [
        ["--solver", "FDNewton", "--solver.n", "8"],
        [
            "--solver",
            "Newton",
            "--solver.scheme",
            "Standard9",
            "--solver.scheme.n",
            "8",
        ],
    ],
    ids=["preset", "composed"],
)
def test_printed_config_and_record_name_the_same_class_paths(tmp_path, capsys, solver):
    args = [*solver, "--problem", "T1"]
    with pytest.raises(SystemExit):
        main(["solve", *args, "--print_config"])
    printed = yaml.safe_load(capsys.readouterr().out)["solver"]
    recorded = _solve(tmp_path, *args)["run"]["solver"]
    assert _class_paths(printed) == _class_paths(recorded)


def _class_paths(config):
    """Every ``class_path`` in a configuration, by its dotted key."""
    paths = {}
    if isinstance(config, dict):
        for key, value in config.items():
            if key == "class_path":
                paths[key] = value
            for sub, path in _class_paths(value).items():
                paths[f"{key}.{sub}"] = path
    return paths


def test_the_run_of_a_record_is_a_config_file(tmp_path):
    record = _solve(
        tmp_path, "--solver", "FDNewton", "--solver.n", "8", "--problem", "T2"
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record["run"]))
    rerun = _solve(tmp_path, "--config", str(path))
    assert rerun["run"] == record["run"]
    assert rerun["metrics"] | {"time": 0} == record["metrics"] | {"time": 0}


def test_python_dash_m_runs_the_command():
    args = ["solve", "--solver", "FDNewton", "--solver.n", "8", "--problem", "T1"]
    out = subprocess.run(
        [sys.executable, "-m", "maesolvers", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("FDNewton on T1: converged")


def _strict(text):
    """``json.loads`` that rejects NaN and Infinity, which strict JSON lacks."""

    def reject(token):
        raise ValueError(f"not strict JSON: {token}")

    return json.loads(text, parse_constant=reject)


def test_a_run_that_does_not_converge_is_reported_and_recorded(tmp_path, capsys):
    args = ["--solver", "FDNewton", "--solver.n", "8", "--solver.max_iter", "1"]
    output = tmp_path / "out.json"
    main(["solve", *args, "--problem", "T1", "--output", str(output)])
    assert capsys.readouterr().out.startswith(
        "FDNewton on T1: did not converge after 1 iterations"
    )
    record = _strict(output.read_text())
    assert record["info"]["converged"] is False
    assert record["info"]["iterations"] == 1


@pytest.fixture
def numpy_solver(tmp_path, monkeypatch):
    """A solver of the user's own whose records hold numpy values and NaN."""
    (tmp_path / "numpy_solver.py").write_text(
        textwrap.dedent(
            """\
            from dataclasses import dataclass
            from typing import ClassVar

            import numpy as np

            from maesolvers import Box, Dirichlet, Solution, Solver
            from maesolvers.core import Trace


            def _zero(x):
                return np.zeros(np.shape(x)[:-1])


            @dataclass(frozen=True, kw_only=True)
            class NumpyRecord(Solver):
                domains: ClassVar[tuple[type, ...]] = (Box,)
                boundaries: ClassVar[tuple[type, ...]] = (Dirichlet,)

                def setup(self, problem):
                    return None

                def run(self, problem, context, callbacks):
                    return Trace(
                        state=None,
                        iterations=np.int64(1),
                        converged=np.bool_(True),
                        history=(np.float64(np.inf),),
                    )

                def finalize(self, problem, context, state):
                    info = {"weights": np.arange(3), "eigmin": np.float32(np.nan)}
                    return Solution(u=_zero, unknowns=np.int64(3), info=info)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("numpy_solver", None)


def test_numpy_values_and_non_finite_floats_are_written_as_strict_json(
    numpy_solver, tmp_path
):
    output = tmp_path / "out.json"
    main(
        [
            "solve",
            "--solver",
            "numpy_solver.NumpyRecord",
            "--problem",
            "T1",
            "--output",
            str(output),
        ]
    )
    info = _strict(output.read_text())["info"]
    assert info["iterations"] == 1
    assert info["converged"] is True
    assert info["history"] == [None]
    assert info["weights"] == [0, 1, 2]
    assert info["eigmin"] is None
