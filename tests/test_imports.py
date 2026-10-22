"""``import maesolvers`` needs numpy and scipy only (ADR 0006); the command needs the cli extra."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("extra", ["jsonargparse", "torch", "lightning", "mlflow"])
def test_import_maesolvers_does_not_import_an_extra(extra):
    code = f"import sys, maesolvers; sys.exit({extra!r} in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_the_command_without_the_cli_extra_exits_with_a_hint(monkeypatch, capsys):
    from maesolvers.cli import main

    monkeypatch.setitem(sys.modules, "jsonargparse", None)  # import fails
    with pytest.raises(SystemExit) as exit_info:
        main(["solve", "--problem", "T1"])
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert message.count("\n") == 1
    assert "jsonargparse" in message
    assert '"maesolvers[cli] @ git+https://github.com/jczhang02/maesolvers"' in message
