"""``import maesolvers`` needs numpy and scipy only (ADR 0006)."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("extra", ["jsonargparse", "torch", "lightning", "mlflow"])
def test_import_maesolvers_does_not_import_an_extra(extra):
    code = f"import sys, maesolvers; sys.exit({extra!r} in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
