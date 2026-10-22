"""Every code block of docs/guide.md runs as written.

A block whose first line is ``# <file name>`` is written to that file in a
scratch directory, which is the working directory and on the import path; any
other ``python`` block runs, in order, in one namespace; every line of an
``sh`` block runs through :func:`maesolvers.cli.main`. A block in any other
language, or in none, fails the test. Without jsonargparse (the ``cli``
extra) the ``sh`` blocks are skipped and a Python block stops at the line
that needs it; every other block still runs, the worked solver's among them.
"""

import importlib.util
import re
import shlex
import sys
from pathlib import Path

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


def _run_python(code, line, namespace, has_cli):
    padded = "\n" * (line - 1) + code  # tracebacks point at the guide's lines
    try:
        exec(compile(padded, str(GUIDE), "exec"), namespace)
    except ImportError as error:  # maesolvers names the missing extra
        if has_cli or "jsonargparse" not in str(error):
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
                _run_python(code, line, namespace, has_cli)
            else:
                pytest.fail(
                    f"guide line {line}: a {language or 'bare'} block cannot run"
                )
        assert {"MyPoisson", "RelaxedPoisson"} <= set(namespace)
        # The worked solver checks its parameters as the shipped solvers do.
        with pytest.raises(TypeError, match="n must be an integer"):
            namespace["MyPoisson"](n=1.5)
        with pytest.raises(ValueError, match="tol must be positive"):
            namespace["MyPoisson"](tol=0.0)
    finally:
        sys.modules.pop("my_methods", None)  # the guide's module, not another test's
