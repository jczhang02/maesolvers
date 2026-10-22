"""Every code block of docs/guide.md runs as written.

A block whose first line is ``# <file name>`` is written to that file in a
scratch directory, which is the working directory and on the import path; any
other ``python`` block runs, in order, in one namespace; every line of an
``sh`` block runs through :func:`maesolvers.cli.main`. Without jsonargparse
(the ``cli`` extra) the blocks that need it are skipped and the rest still
run, the worked solver among them.
"""

import importlib.util
import re
import shlex
import sys
from pathlib import Path

GUIDE = Path(__file__).parents[1] / "docs" / "guide.md"
FENCE = re.compile(r"^```(\w+)\n(.*?)^```$", re.M | re.S)
FILE_NAME = re.compile(r"# ([\w.-]+\.(?:py|yaml|json))\n")
COMMAND = re.compile(r"^(?:python -m )?maesolvers (.*)$")


def _blocks():
    """(language, code, line of the code's first line in the guide) per block."""
    text = GUIDE.read_text()
    for match in FENCE.finditer(text):
        line = text.count("\n", 0, match.start(2)) + 1
        yield match.group(1), match.group(2), line


def _run_python(code, line, namespace):
    padded = "\n" * (line - 1) + code  # tracebacks point at the guide's lines
    exec(compile(padded, str(GUIDE), "exec"), namespace)


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
            needs_cli = language == "sh" or "from_config" in code
            if file_name:
                (tmp_path / file_name.group(1)).write_text(code)
            elif needs_cli and not has_cli:
                continue
            elif language == "sh":
                _run_commands(code)
            elif language == "python":
                _run_python(code, line, namespace)
    finally:
        sys.modules.pop("my_methods", None)  # the guide's module, not another test's
    assert {"MyPoisson", "RelaxedPoisson"} <= set(namespace)
