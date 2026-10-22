"""The discrete solutions of the presets, pinned to the numbers of 97fad98.

Commit 97fad98 is the last one before FDNewton and FDPoisson became a scheme
under an iteration (ADR 0004). These numbers are what it computed, so a later
refactor that moves a discrete solution fails here. The iteration counts are
exact; the values allow 1e-9 relative, for rounding differences between
machines and library builds (``benchmarks/standard_fd.md``, Finding 4), far
below any change to the scheme or the iteration.
"""

import numpy as np
import pytest

from maesolvers import T1, T2, FDNewton, FDPoisson
from maesolvers.metrics import evaluate

NODES = [(4, 4), (8, 8), (12, 12), (15, 15)]
"""Grid indices ``(i, j)`` of the pinned values; n = 16 cells, h = 1/16."""

PINNED = {
    (FDNewton, "T1"): (
        3,
        2.664328292198981e-4,
        [
            1.0646246239525838,
            1.2842880255683313,
            1.7552287365029977,
            2.4082806232534213,
        ],
    ),
    (FDNewton, "T2"): (
        6,
        2.211284566996774e-2,
        [
            -1.3693212657951757,
            -1.224962820943276,
            -0.9369961733162981,
            -0.5142383377957059,
        ],
    ),
    (FDPoisson, "T1"): (
        47,
        2.6643282921368083e-4,
        [1.0646246239525687, 1.2842880255683207, 1.7552287365015355, 2.408280623253696],
    ),
    (FDPoisson, "T2"): (
        52,
        2.2112845669807868e-2,
        [
            -1.369321265795172,
            -1.2249628209432628,
            -0.9369961733162891,
            -0.5142383377955461,
        ],
    ),
}


@pytest.mark.parametrize(
    ("solver", "problem"), [(s, p) for s in (FDNewton, FDPoisson) for p in (T1, T2)]
)
def test_presets_keep_their_discrete_solutions(solver, problem):
    iterations, max_error, values = PINNED[solver, problem.name]
    result = solver(n=16).solve(problem)
    flat = [i * 17 + j for i, j in NODES]
    np.testing.assert_allclose(result.nodes[flat], np.array(NODES) / 16)
    assert result.info["iterations"] == iterations
    np.testing.assert_allclose(evaluate(result)["max"], max_error, rtol=1e-9)
    np.testing.assert_allclose(result.values[flat], values, rtol=1e-9)
