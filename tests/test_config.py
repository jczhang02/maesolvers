"""Solvers from a configuration: ``maesolvers.from_config`` (ADR 0005)."""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import sys
import textwrap
import typing

import pytest

import maesolvers
from maesolvers import (
    SOLVERS,
    T1,
    FDNewton,
    FDPoisson,
    Newton,
    PoissonIteration,
    Solver,
    Standard9,
    from_config,
)
from maesolvers.classical import Scheme
from maesolvers.neural import Loss, Network, NeuralSolver, Optimizer, Sampler

pytest.importorskip("jsonargparse")

PRESET = {"class_path": "FDPoisson", "init_args": {"n": 16}}
COMPOSED = {
    "class_path": "Newton",
    "init_args": {
        "max_iter": 7,
        "scheme": {"class_path": "Standard9", "init_args": {"n": 16}},
    },
}


def test_a_dict_builds_the_solver_of_the_same_keyword_arguments():
    assert from_config(PRESET) == FDPoisson(n=16)


def test_a_dict_builds_nested_parts():
    assert from_config(COMPOSED) == Newton(scheme=Standard9(n=16), max_iter=7)


@pytest.mark.parametrize("config", [PRESET, COMPOSED], ids=["preset", "composed"])
@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_file_builds_what_the_dict_builds(tmp_path, config, suffix):
    path = tmp_path / f"solver{suffix}"
    path.write_text(json.dumps(config))  # JSON is YAML too
    assert from_config(path) == from_config(config)
    assert from_config(str(path)) == from_config(config)


def test_yaml_with_a_float_in_exponent_notation(tmp_path):
    # YAML 1.1 reads 1e-10 as a string; the signature makes it a float.
    path = tmp_path / "solver.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            class_path: FDNewton
            init_args:
              n: 16
              tol: 1e-10
            """
        )
    )
    assert from_config(path) == FDNewton(n=16, tol=1e-10)


def test_init_args_with_no_value_in_yaml_means_the_defaults(tmp_path):
    path = tmp_path / "solver.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            class_path: Newton
            init_args:
              scheme:
                class_path: Standard9
                init_args:
            """
        )
    )
    assert from_config(path) == Newton(scheme=Standard9())
    assert from_config({"class_path": "FDNewton", "init_args": None}) == FDNewton()


def test_short_full_and_recorded_class_paths_build_equal_solvers():
    paths = ["FDPoisson", "maesolvers.FDPoisson", FDPoisson(n=16).hparams["class_path"]]
    built = {from_config({**PRESET, "class_path": path}) for path in paths}
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
def test_a_result_reruns_from_its_hparams(solver):
    result = solver.solve(T1)
    rebuilt = from_config(result.hparams)
    assert rebuilt == solver
    assert (rebuilt.solve(result.problem).values == result.values).all()
    assert from_config(json.loads(json.dumps(result.hparams))) == solver


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_preset_builds_from_its_name_with_defaults(name):
    assert from_config({"class_path": name}) == SOLVERS[name]()


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
        "class_path": "my_methods.DampedNewton",
        "init_args": {
            "damping": "5e-1",
            "scheme": {"class_path": "my_methods.Coarse", "init_args": {"n": 8}},
        },
    }
    solver = from_config(config)
    assert solver == user_module.DampedNewton(
        damping=0.5, scheme=user_module.Coarse(n=8)
    )
    assert from_config(solver.hparams) == solver


# A parameter whose annotation jsonargparse cannot resolve takes any value
# unchecked, so these guard that every solver and scheme class maesolvers
# exports, and a user's subclass in its own module, keep annotations it can
# resolve. The classes come from maesolvers.__all__, so a new one is checked
# on arrival.


def _exported(base):
    classes = (getattr(maesolvers, name) for name in maesolvers.__all__)
    return sorted(
        cls.__name__
        for cls in classes
        if inspect.isclass(cls)
        and issubclass(cls, base)
        and not inspect.isabstract(cls)
    )


SOLVER_CLASSES = _exported(Solver)
SCHEME_CLASSES = _exported(Scheme)
NEURAL_SLOTS = {
    "network": (Network, {"class_path": "MLP"}),
    "loss": (Loss, {"class_path": "PenaltyLoss"}),
    "sampler": (Sampler, {"class_path": "UniformSampler"}),
    "optimizer": (Optimizer, {"class_path": "LBFGS"}),
}
NEURAL_PART_CLASSES = [
    (slot, name) for slot, (base, _) in NEURAL_SLOTS.items() for name in _exported(base)
]


def test_the_guards_see_every_exported_solver_and_scheme():
    assert set(SOLVER_CLASSES) >= {
        "FDNewton",
        "FDPoisson",
        "Newton",
        "PoissonIteration",
    }
    assert set(SCHEME_CLASSES) >= {"Standard9"}
    assert {name for _, name in NEURAL_PART_CLASSES} >= {
        "MLP",
        "HardDirichlet",
        "PenaltyLoss",
        "UniformSampler",
        "LBFGS",
        "Adam",
    }


def _class(class_path, user_module):
    module, _, name = class_path.rpartition(".")
    return getattr(user_module if module else maesolvers, name)


def _numeric_parameters(cls):
    hints = typing.get_type_hints(cls)
    numeric = (int, float, int | None, float | None)
    return [
        f.name for f in dataclasses.fields(cls) if f.init and hints[f.name] in numeric
    ]


def _scheme_parts(cls):
    """A default part for every field of ``cls`` that takes a scheme (Filtered's)."""
    hints = typing.get_type_hints(cls)
    return {
        f.name: {"class_path": "Monotone17"}
        for f in dataclasses.fields(cls)
        if f.init and hints[f.name] is Scheme
    }


def _solver(cls, class_path, **init_args):
    """A configuration of ``cls`` with a default part in every required part slot."""
    fields = {f.name for f in dataclasses.fields(cls) if f.init}
    if "scheme" in fields:
        init_args.setdefault("scheme", {"class_path": "Standard9"})
    for slot, (_, default) in NEURAL_SLOTS.items():
        if slot in fields:
            init_args.setdefault(slot, default)
    return {"class_path": class_path, "init_args": init_args}


@pytest.mark.parametrize("class_path", [*SOLVER_CLASSES, "my_methods.DampedNewton"])
def test_every_numeric_solver_parameter_is_checked_against_its_type(
    user_module, class_path
):
    cls = _class(class_path, user_module)
    names = _numeric_parameters(cls)
    assert names
    assert type(from_config(_solver(cls, class_path))) is cls
    for name in names:
        with pytest.raises(ValueError, match=f'key "{name}"'):
            from_config(_solver(cls, class_path, **{name: "many"}))


@pytest.mark.parametrize("scheme", [*SCHEME_CLASSES, "my_methods.Coarse"])
def test_every_numeric_scheme_parameter_is_checked_against_its_type(
    user_module, scheme
):
    cls = _class(scheme, user_module)
    names = _numeric_parameters(cls)
    assert names
    for name in names:
        part = {"class_path": scheme, "init_args": {**_scheme_parts(cls), name: "many"}}
        with pytest.raises(ValueError, match=f'key "{name}"'):
            from_config(_solver(Newton, "Newton", scheme=part))


@pytest.mark.parametrize(("slot", "part"), NEURAL_PART_CLASSES)
def test_every_numeric_neural_part_parameter_is_checked_against_its_type(slot, part):
    names = _numeric_parameters(getattr(maesolvers, part))
    assert names
    for name in names:
        config = _solver(getattr(maesolvers, part), part, **{name: "many"})
        with pytest.raises(ValueError, match=f'key "{name}"'):
            from_config(_solver(NeuralSolver, "NeuralSolver", **{slot: config}))


def _with_init_args(**init_args):
    return {"class_path": "FDPoisson", "init_args": init_args}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({**PRESET, "init_arg": {}}, "init_arg"),
        (_with_init_args(nn=16), "'nn'"),
        (
            {
                "class_path": "Newton",
                "init_args": {
                    "scheme": {"class_path": "Standard9", "init_args": {"m": 8}}
                },
            },
            "'m'",
        ),
        ({"class_path": "Newton"}, "scheme"),
        (
            {"class_path": "Newton", "init_args": {"scheme": {"init_args": None}}},
            "scheme",
        ),
        ({"class_path": "FDPoison"}, "FDPoison"),
        ({"class_path": "Standard9"}, "Standard9"),
        ({"init_args": {"n": 16}}, "class_path"),
        (_with_init_args(n=2.5), 'key "n"'),
    ],
    ids=[
        "top-level key",
        "init arg",
        "nested init arg",
        "missing init arg",
        "init_args with no class_path",
        "unknown class",
        "not a solver",
        "no class_path",
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
    with pytest.raises(FileNotFoundError, match=re.escape("nothing.yaml")):
        from_config(tmp_path / "nothing.yaml")
