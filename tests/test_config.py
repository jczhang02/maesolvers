"""Solvers and problems from a configuration: ``maesolvers.from_config`` (ADR 0005)."""

from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
import typing

import pytest

from maesolvers import (
    SOLVERS,
    T1,
    T2,
    FDNewton,
    FDPoisson,
    Newton,
    PoissonIteration,
    Standard9,
    from_config,
)

pytest.importorskip("jsonargparse")

PRESET = {
    "solver": {"class_path": "FDPoisson", "init_args": {"n": 16}},
    "problem": "T2",
}
COMPOSED = {
    "solver": {
        "class_path": "Newton",
        "init_args": {
            "max_iter": 7,
            "scheme": {"class_path": "Standard9", "init_args": {"n": 16}},
        },
    },
    "problem": "T1",
}


def test_a_dict_builds_the_solver_of_the_same_keyword_arguments():
    assert from_config(PRESET) == (FDPoisson(n=16), T2)


def test_a_dict_builds_nested_parts():
    assert from_config(COMPOSED) == (Newton(scheme=Standard9(n=16), max_iter=7), T1)


@pytest.mark.parametrize("config", [PRESET, COMPOSED], ids=["preset", "composed"])
@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_file_builds_what_the_dict_builds(tmp_path, config, suffix):
    path = tmp_path / f"run{suffix}"
    path.write_text(json.dumps(config))  # JSON is YAML too
    assert from_config(path) == from_config(config)
    assert from_config(str(path)) == from_config(config)


def test_yaml_with_a_float_in_exponent_notation(tmp_path):
    # YAML 1.1 reads 1e-10 as a string; the signature makes it a float.
    path = tmp_path / "run.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            solver:
              class_path: FDNewton
              init_args:
                n: 16
                tol: 1e-10
            problem: T1
            """
        )
    )
    assert from_config(path) == (FDNewton(n=16, tol=1e-10), T1)


def test_short_full_and_recorded_class_paths_build_equal_solvers():
    paths = ["FDPoisson", "maesolvers.FDPoisson", FDPoisson(n=16).hparams["class_path"]]
    built = {
        from_config({**PRESET, "solver": {**PRESET["solver"], "class_path": path}})[0]
        for path in paths
    }
    assert built == {FDPoisson(n=16)}


@pytest.mark.parametrize(
    "solver",
    [
        FDNewton(n=8, tol=1e-9),
        FDPoisson(n=10, spline_degree=2),
        Newton(scheme=Standard9(n=8), max_iter=7),
        PoissonIteration(scheme=Standard9(n=12), tol=1e-9),
    ],
    ids=["FDNewton", "FDPoisson", "Newton", "PoissonIteration"],
)
def test_a_result_reruns_from_its_record(solver):
    result = solver.solve(T1)
    record = {"solver": result.hparams, "problem": result.problem.name}
    rebuilt, problem = from_config(json.loads(json.dumps(record)))
    assert rebuilt == solver and problem is T1
    assert (rebuilt.solve(problem).values == result.values).all()


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_preset_builds_from_its_name_with_defaults(name):
    assert from_config({"solver": {"class_path": name}, "problem": "T1"})[0] == (
        SOLVERS[name]()
    )


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    """A module of the user's own, outside maesolvers, with postponed annotations."""
    (tmp_path / "my_methods.py").write_text(
        textwrap.dedent(
            """\
            from __future__ import annotations

            from dataclasses import dataclass

            from maesolvers import Newton, Standard9


            @dataclass(frozen=True, kw_only=True)
            class Coarse(Standard9):
                pass


            @dataclass(frozen=True, kw_only=True)
            class DampedNewton(Newton):
                damping: float = 1.0
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import my_methods

    yield my_methods
    sys.modules.pop("my_methods", None)


def test_a_users_own_solver_and_scheme_build_from_their_class_paths(user_module):
    config = {
        "solver": {
            "class_path": "my_methods.DampedNewton",
            "init_args": {
                "damping": "5e-1",
                "scheme": {"class_path": "my_methods.Coarse", "init_args": {"n": 8}},
            },
        },
        "problem": "T1",
    }
    solver, _ = from_config(config)
    assert solver == user_module.DampedNewton(
        damping=0.5, scheme=user_module.Coarse(n=8)
    )
    assert from_config({"solver": solver.hparams, "problem": "T1"})[0] == solver


def _numeric_parameters(cls):
    hints = typing.get_type_hints(cls)
    return [
        f.name
        for f in dataclasses.fields(cls)
        if f.init and hints[f.name] in (int, float)
    ]


def _run(class_path, **init_args):
    if class_path not in SOLVERS:
        init_args.setdefault("scheme", {"class_path": "Standard9"})
    return {
        "solver": {"class_path": class_path, "init_args": init_args},
        "problem": "T1",
    }


# A parameter whose annotation jsonargparse cannot resolve takes any value
# unchecked, so these guard that solver and scheme classes, a user's subclass
# in its own module included, keep annotations it can resolve.


@pytest.mark.parametrize(
    "class_path",
    [*sorted(SOLVERS), "Newton", "PoissonIteration", "my_methods.DampedNewton"],
)
def test_every_numeric_solver_parameter_is_checked_against_its_type(
    user_module, class_path
):
    solver, _ = from_config(_run(class_path))
    names = _numeric_parameters(type(solver))
    assert {"max_iter", "tol"} <= set(names)
    for name in names:
        with pytest.raises(ValueError, match=f'key "{name}"'):
            from_config(_run(class_path, **{name: "many"}))


@pytest.mark.parametrize("scheme", ["Standard9", "my_methods.Coarse"])
def test_every_numeric_scheme_parameter_is_checked_against_its_type(
    user_module, scheme
):
    for name in _numeric_parameters(Standard9):
        part = {"class_path": scheme, "init_args": {name: "many"}}
        with pytest.raises(ValueError, match=f'key "{name}"'):
            from_config(_run("Newton", scheme=part))


def _with_init_args(**init_args):
    return {
        "solver": {"class_path": "FDPoisson", "init_args": init_args},
        "problem": "T1",
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({**PRESET, "problme": "T1"}, "problme"),
        (_with_init_args(nn=16), "'nn'"),
        (
            {
                "solver": {
                    "class_path": "Newton",
                    "init_args": {
                        "scheme": {"class_path": "Standard9", "init_args": {"m": 8}}
                    },
                },
                "problem": "T1",
            },
            "'m'",
        ),
        ({"solver": {"class_path": "FDPoison"}, "problem": "T1"}, "FDPoison"),
        ({"solver": {"class_path": "Standard9"}, "problem": "T1"}, "Standard9"),
        ({**PRESET, "problem": "T99"}, "T99"),
        ({"solver": PRESET["solver"]}, "problem"),
        ({"problem": "T1"}, "solver"),
        (_with_init_args(n=2.5), 'key "n"'),
    ],
    ids=[
        "top-level key",
        "init arg",
        "nested init arg",
        "unknown class",
        "not a solver",
        "unknown problem",
        "no problem",
        "no solver",
        "wrong type",
    ],
)
def test_a_wrong_or_missing_key_is_a_value_error_that_names_it(config, message):
    with pytest.raises(ValueError, match=message):
        from_config(config)


@pytest.mark.parametrize(
    ("init_args", "error", "message"),
    [
        ({"n": 1}, ValueError, "n must be at least 2, got 1"),
        ({"tol": -1.0}, ValueError, "tol must be positive"),
    ],
)
def test_a_bad_value_raises_what_the_constructor_raises(init_args, error, message):
    with pytest.raises(error, match=message):
        FDPoisson(**init_args)
    with pytest.raises(error, match=message):
        from_config(_with_init_args(**init_args))


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.yaml"):
        from_config(tmp_path / "nothing.yaml")
