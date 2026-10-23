"""Published numbers the solvers must match.

Source: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column
"Standard" (same values in Froese-Oberman, SIAM J. Numer. Anal. 49 (2011),
Table 3, column "Standard F.D."). Max error at the grid nodes on an N x N
grid with h = 1 / N; see benchmarks/standard_fd.md for the convention.

Benamou-Froese-Oberman, J. Comput. Phys. 260 (2014), Table 1: max error of
the gradient map on T16. The paper states its N as nodes per side, "the
spatial step size dx coming from the number of grid points N_X along each
dimension" (Section 5.2.1, p. 117), so h = 1/(N - 1); FDTransport prints the
column at every N there, which confirms it (benchmarks/bfo2014.md, Finding 1).
N = 32 and 64 are pinned.

Benamou-Froese-Oberman, ESAIM: M2AN 44 (2010), Table 7, column "(M1), (M2)":
u_min of T7 on [-1, 1]^2, where N counts nodes per side, so N = 21 is n = 20
cells and the origin is a node (benchmarks/standard_fd.md, Finding 3).
FDPoisson matches it at every printed N (benchmarks/t7_bfo2010.csv); the first
three are pinned here. FDNewton does not (a non-convex root from N = 41), so
it is not pinned. T7 is also the one pinned problem on a box whose side is not
1 and whose lower corner is not the origin.
"""

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from centred import C2_CENTRED
from scipy.sparse.linalg import spsolve

from maesolvers import (
    CATALOGUE,
    T1,
    T2,
    T7,
    T9,
    T16,
    T20,
    FDFiltered,
    FDHybrid,
    FDMonotone,
    FDNewton,
    FDPoisson,
    FDTransport,
    Newton,
    Problem,
)
from maesolvers.metrics import evaluate, gradient_fd, gradient_map_error

PUBLISHED_T1 = {31: 7.14e-5, 63: 1.73e-5}
PUBLISHED_T2 = {31: 17.15e-3, 63: 12.53e-3}
PUBLISHED_T7 = {20: 0.2892, 40: 0.2734, 60: 0.2682}  # key: n = N - 1 cells


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_T1.items()))
@pytest.mark.parametrize("solver", [FDNewton, FDPoisson])
def test_smooth_example_matches_froese_oberman(solver, n, expected):
    err = evaluate(solver(n=n).solve(T1))["max"]
    assert err == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_T2.items()))
def test_blowup_example_matches_froese_oberman_with_poisson_iteration(n, expected):
    err = evaluate(FDPoisson(n=n).solve(T2))["max"]
    assert err == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_T7.items()))
def test_t7_minimum_matches_benamou_froese_oberman_with_poisson_iteration(n, expected):
    # Table 7 prints four decimals; n = 60 gives 0.268153, so compare rounded.
    assert round(evaluate(FDPoisson(n=n).solve(T7))["u_origin"], 4) == expected


# Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 3, columns "9 Point",
# "17 Point", "33 Point": max error at the nodes of the monotone scheme, h = 1/N.
# Matched to the printed digits on T2 at every printed N and on T1 from N = 127
# on (benchmarks/monotone17.md); the cheap ones are pinned here.
SINUM_T2 = {  # N: (9, 17, 33 points), in units of 1e-3
    31: (1.7, 1.7, 1.7),
    63: (0.9, 0.6, 0.6),
}
SINUM_T1_127 = {9: 15.9, 17: 4.6, 33: 1.8}  # units of 1e-4


@pytest.mark.parametrize("n", sorted(SINUM_T2))
@pytest.mark.parametrize("column", range(3))
def test_monotone_blowup_example_matches_the_sinum_table(n, column):
    stencil = (9, 17, 33)[column]
    err = evaluate(FDMonotone(n=n, stencil=stencil).solve(T2))["max"]
    assert round(err / 1e-3, 1) == SINUM_T2[n][column]


@pytest.mark.parametrize("stencil", sorted(SINUM_T1_127))
def test_monotone_smooth_example_matches_the_sinum_table_at_127(stencil):
    err = evaluate(FDMonotone(n=127, stencil=stencil).solve(T1))["max"]
    assert round(err / 1e-4, 1) == SINUM_T1_127[stencil]


# Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Monotone"
# (17 points). Ours agrees to the printed digits on T1 from N = 181 on and on
# T2 from N = 89 on (benchmarks/monotone17.md); the coarser N are order of
# magnitude only and not pinned.
JCP_T1_181 = 44.65  # units of 1e-5
JCP_T2 = {89: 0.37, 127: 0.35}  # units of 1e-3


def test_monotone_smooth_example_matches_the_jcp_table_at_181():
    err = evaluate(FDMonotone(n=181).solve(T1))["max"]
    assert round(err / 1e-5, 2) == JCP_T1_181


@pytest.mark.parametrize("n,expected", sorted(JCP_T2.items()))
def test_monotone_blowup_example_matches_the_jcp_table(n, expected):
    err = evaluate(FDMonotone(n=n).solve(T2))["max"]
    assert round(err / 1e-3, 2) == expected


# Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Hybrid" (17
# points). Ours (FDHybrid, h = 1/N) agrees to the printed digits on T2 from
# N = 127 on, and at every N with h = 1/(N - 1) (benchmarks/hybrid.md,
# Finding 2); T1, T20 and the cone are a mismatch and not pinned.
JCP_HYBRID_T2 = {127: 0.20, 181: 0.12}  # units of 1e-3


@pytest.mark.parametrize("n,expected", sorted(JCP_HYBRID_T2.items()))
def test_hybrid_blowup_example_matches_the_jcp_table(n, expected):
    err = evaluate(FDHybrid(n=n).solve(T2))["max"]
    assert round(err / 1e-3, 2) == expected


# Froese-Oberman, SIAM J. Numer. Anal. 51 (2013), Tables 4.1 and 4.2. Their N
# counts nodes per side, so n = N - 1 cells (benchmarks/filtered.md, Finding 1),
# and their C^2 example is T1 centred at (1/2, 1/2), which the catalogue lacks.
# Only converged solves in exact columns (ADR 0002) are pinned. On T2
# FDFiltered's damped Newton stalls; with the paper's undamped iteration
# (p. 438) the 9-point filtered column of Table 4.1 is exact (Finding 4), so it
# is pinned with a test-local copy of run_filtered.UndampedNewton.


# (N, stencil): units of 1e-5. The 17-point cells are Table 4.2's filtered
# column, which is exact. N = 63: 1.06; Table 4.1 prints 1.04 there for every
# stencil, read as a misprint (Finding 2), so its three filtered columns are
# graded order of magnitude and 9 and 33 points are not pinned at N = 63. They
# are pinned at N = 31: the filter keeps the standard scheme at every node
# (monotone_fraction 0), so each stencil solves the standard scheme of the exact
# Table 4.2 column, 4.54, and the pins check that 9 and 33 points do too.
FO2013_FILTERED_C2 = {(31, 9): 4.54, (31, 17): 4.54, (31, 33): 4.54, (63, 17): 1.06}
FO2013_STANDARD = {  # Table 4.2, column "Standard": (problem, N, units, value)
    "C2": (C2_CENTRED, 31, 1e-5, 4.54),
    "T20": (T20, 31, 1e-4, 3.78),
    "T2-31": (T2, 31, 1e-3, 17.38),
    "T2-63": (T2, 63, 1e-3, 12.62),
}
FO2013_MONOTONE = {  # Table 4.1, columns "Monotone": (problem, N, stencil, units, value)
    "C2-17": (C2_CENTRED, 31, 17, 1e-5, 9.12),
    "C2-17-63": (C2_CENTRED, 63, 17, 1e-5, 3.42),
    "T2-9": (T2, 63, 9, 1e-3, 0.86),
    "T2-17": (T2, 63, 17, 1e-3, 0.59),
    "T2-33": (T2, 63, 33, 1e-3, 0.59),
}


@pytest.mark.parametrize(("N", "stencil"), sorted(FO2013_FILTERED_C2))
def test_filtered_smooth_example_matches_froese_oberman_2013(N, stencil):
    result = FDFiltered(n=N - 1, stencil=stencil).solve(C2_CENTRED)
    assert result.info["converged"]
    expected = FO2013_FILTERED_C2[N, stencil]
    assert round(evaluate(result)["max"] / 1e-5, 2) == expected


@pytest.mark.parametrize("case", sorted(FO2013_STANDARD))
def test_standard_column_of_froese_oberman_2013_has_n_minus_one_cells(case):
    problem, N, units, expected = FO2013_STANDARD[case]
    err = evaluate(FDPoisson(n=N - 1).solve(problem))["max"]
    assert round(err / units, 2) == expected


@pytest.mark.parametrize("case", sorted(FO2013_MONOTONE))
def test_monotone_columns_of_froese_oberman_2013(case):
    problem, N, stencil, units, expected = FO2013_MONOTONE[case]
    err = evaluate(FDMonotone(n=N - 1, stencil=stencil).solve(problem))["max"]
    assert round(err / units, 2) == expected


@dataclasses.dataclass(frozen=True, kw_only=True)
class UndampedNewton(Newton):
    """The full Newton step, as run_filtered.UndampedNewton (FO 2013, p. 438)."""

    def step(
        self, problem: Problem, context: Any, iterate: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        r = self._scheme.residual(context, iterate)
        norm = float(np.max(np.abs(r)))
        if self.converged(problem, context, norm):
            return iterate, norm
        candidate = iterate + spsolve(self._scheme.jacobian(context, iterate), -r)
        new_norm = float(np.max(np.abs(self._scheme.residual(context, candidate))))
        return (candidate, new_norm) if math.isfinite(new_norm) else None


# Table 4.1, T2, 9-point filtered column, units of 1e-3: exact with the undamped
# iteration (74 and 173 steps). The 17- and 33-point columns are a mismatch (no
# convergence from N = 127) and are not pinned; at N = 31 and 63 they would not
# discriminate anyway, since there every filtered solve and the 17- and 33-point
# monotone scheme give the same error. At N = 63 the 9-point monotone scheme
# gives 0.86, so this pin tells the filtered scheme from it.
FO2013_FILTERED_T2_9 = {31: 1.74, 63: 0.59}


@pytest.mark.parametrize(("N", "expected"), sorted(FO2013_FILTERED_T2_9.items()))
def test_filtered_t2_matches_froese_oberman_2013_with_their_undamped_newton(
    N, expected
):
    scheme = FDFiltered(n=N - 1, stencil=9).scheme
    result = UndampedNewton(scheme=scheme, max_iter=600).solve(T2)
    assert result.info["converged"]
    assert round(evaluate(result)["max"] / 1e-3, 2) == expected


BFO2014_T16 = {32: 0.0220, 64: 0.0110}  # Table 1, max error; key: N nodes per side


@pytest.mark.parametrize(("N", "expected"), sorted(BFO2014_T16.items()))
def test_transport_scheme_matches_bfo_2014_table_1(N, expected):
    result = FDTransport(n=N - 1).solve(T16)
    assert result.info["converged"]
    points, h = result.nodes[result.interior], result.info["h"][0]
    error = gradient_map_error(
        gradient_fd(result.u, points, step=h), gradient_fd(T16.exact, points)
    )
    assert round(error["gradient_max"], 4) == expected


# Froese-Oberman, SIAM J. Numer. Anal. 49 (2011), Table 4: max error of the
# 19-point scheme in 3D, with h = 1/N (ADR 0001). T9 is order of magnitude
# (benchmarks/monotone19.md) and agrees to the printed digits at N = 21 and 31;
# N = 31 takes a minute, so only N = 21 is pinned. The cells that agree with
# h = 1/(N - 1) are not pinned: the paper used one convention, and which one
# its N follows stays open (Finding 1).
SINUM_TABLE4_EXACT = [  # (problem, n cells, printed max error in units of 1e-4)
    (T9, 21, 127),
]


@pytest.mark.parametrize(("problem", "n", "expected"), SINUM_TABLE4_EXACT)
def test_the_19_point_scheme_matches_sinum_table_4(problem, n, expected):
    result = FDMonotone(n=n, stencil=19).solve(problem)
    assert result.info["converged"]
    assert round(evaluate(result)["max"] / 1e-4) == expected


def _load_runner():
    """benchmarks/run_monotone19.py, for its test-local problem C1_3D."""
    path = Path(__file__).parent.parent / "benchmarks" / "run_monotone19.py"
    spec = importlib.util.spec_from_file_location("run_monotone19", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # its dataclasses look their module up in sys.modules
    with mock.patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


def test_the_3d_c1_example_solves_det_d2u_equals_f():
    """Example (15) of the paper: f is the determinant of u's Hessian off the ball."""
    c1 = _load_runner().C1_3D
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, (200, 3))
    r = np.linalg.norm(x - 0.5, axis=1)
    x = x[np.abs(r - 0.2) > 0.02]
    e, step = np.eye(3), 1e-4
    hessian = np.empty((len(x), 3, 3))
    for i in range(3):
        for j in range(3):
            hessian[:, i, j] = (
                c1.exact(x + step * (e[i] + e[j]))
                - c1.exact(x + step * (e[i] - e[j]))
                - c1.exact(x - step * (e[i] - e[j]))
                + c1.exact(x - step * (e[i] + e[j]))
            ) / (4 * step**2)
    np.testing.assert_allclose(np.linalg.det(hessian), c1.f(x), atol=1e-6)
    r = np.linalg.norm(x - 0.5, axis=1)
    np.testing.assert_allclose(c1.f(x), np.where(r > 0.2, 1 - 0.4 / r + 0.04 / r**2, 0))
    assert c1.dim == 3 and c1.name not in CATALOGUE
