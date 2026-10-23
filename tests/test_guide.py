"""Every code block of docs/guide.md runs as written.

A block whose first line is ``# <file name>`` is written to that file in a
scratch directory, which is the working directory and on the import path; any
other ``python`` block runs, in order, in one namespace; every line of an
``sh`` block runs through :func:`maesolvers.cli.main`. A block in any other
language, or in none, fails the test. Without jsonargparse (the ``cli``
extra) the ``sh`` blocks are skipped and a Python block stops at the line
that needs it, and so does a block without torch or lightning (the
``neural`` extra); every other block still runs, the worked solver's among
them.
"""

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

GUIDE = Path(__file__).parents[1] / "docs" / "guide.md"
FENCE = re.compile(r"^```(\w*)\n(.*?)^```$", re.M | re.S)
FILE_NAME = re.compile(r"# ([\w.-]+\.(?:py|yaml|json))\n")
COMMAND = re.compile(r"^(?:python -m )?maesolvers (.*)$")


def _blocks():
    """(language, code, line of the code's first line in the guide) per block."""
    text = GUIDE.read_text()
    for match in FENCE.finditer(text):
        line = text.count("\n", 0, match.start(2)) + 1
        yield match.group(1), match.group(2), line


def _run_python(code, line, namespace, has_cli, has_neural):
    padded = "\n" * (line - 1) + code  # tracebacks point at the guide's lines
    try:
        exec(compile(padded, str(GUIDE), "exec"), namespace)
    except ImportError as error:  # maesolvers names the missing extra
        missing_cli = not has_cli and "jsonargparse" in str(error)
        missing_neural = not has_neural and re.search(r"torch|neural", str(error))
        if not (missing_cli or missing_neural):
            raise


def _run_commands(code):
    from maesolvers.cli import main

    for command in code.splitlines():
        match = COMMAND.match(command)
        assert match, f"not a maesolvers command: {command}"
        main(shlex.split(match.group(1)))


def test_every_block_of_the_guide_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    has_cli = importlib.util.find_spec("jsonargparse") is not None
    has_neural = all(
        importlib.util.find_spec(name) is not None for name in ("torch", "lightning")
    )
    namespace = {"__name__": "guide"}
    try:
        for language, code, line in _blocks():
            file_name = FILE_NAME.match(code)
            if file_name:
                (tmp_path / file_name.group(1)).write_text(code)
                importlib.invalidate_caches()  # a failed import may have listed tmp_path
            elif language == "sh":
                if has_cli:
                    _run_commands(code)
            elif language == "python":
                _run_python(code, line, namespace, has_cli, has_neural)
            else:
                pytest.fail(
                    f"guide line {line}: a {language or 'bare'} block cannot run"
                )
        assert {"MyPoisson", "RelaxedPoisson"} <= set(namespace)
        # The worked solver and iteration check their parameters as the shipped
        # solvers do, with the same messages.
        my_poisson, relaxed = namespace["MyPoisson"], namespace["RelaxedPoisson"]
        assert type(my_poisson(n=np.int64(16)).n) is int
        assert type(my_poisson(tol=np.float32(1e-3)).tol) is float
        with pytest.raises(TypeError, match="n must be an integer"):
            my_poisson(n=1.5)
        with pytest.raises(ValueError, match="n must be at least 2"):
            my_poisson(n=1)
        with pytest.raises(TypeError, match="tol must be a real number"):
            my_poisson(tol="1e-3")
        for tol in (0.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="tol must be positive and finite"):
                my_poisson(tol=tol)
        scheme = namespace["Standard9"](n=8)
        assert type(relaxed(scheme=scheme, omega=np.float64(0.8)).omega) is float
        with pytest.raises(TypeError, match="omega must be a real number"):
            relaxed(scheme=scheme, omega="0.5")
        with pytest.raises(ValueError, match="omega must be at most 1"):
            relaxed(scheme=scheme, omega=1.5)
        with pytest.raises(ValueError, match="omega must be positive and finite"):
            relaxed(scheme=scheme, omega=0.0)
        if has_neural:  # the worked optimizer part checks its parameters too
            momentum = namespace["Momentum"]
            assert type(momentum(lr=np.float64(0.1)).lr) is float
            with pytest.raises(ValueError, match="momentum must be below 1"):
                momentum(momentum=1.0)
            with pytest.raises(ValueError, match="lr must be positive and finite"):
                momentum(lr=0.0)
    finally:
        for module in ("my_methods", "my_parts"):  # the guide's, not another test's
            sys.modules.pop(module, None)


def test_the_guide_builds_torch_lbfgs_with_one_iteration_per_step():
    # torch.optim.LBFGS makes up to 20 iterations per step() by default, which
    # would break the training loop's count of steps
    text = GUIDE.read_text(encoding="utf-8")
    assert "`max_iter=1`, `tolerance_grad=0` and `tolerance_change=0`" in text
