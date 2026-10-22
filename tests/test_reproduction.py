"""Published numbers the solvers must match.

Source: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column
"Standard" (same values in Froese-Oberman, SIAM J. Numer. Anal. 49 (2011),
Table 3, column "Standard F.D."). Max error at the grid nodes on an N x N
grid with h = 1 / N; see benchmarks/standard_fd.md for the convention.

Benamou-Froese-Oberman, ESAIM: M2AN 44 (2010), Table 7: u_min of T7 on
[-1, 1]^2, where N counts nodes per side, so N = 21 is n = 20 cells and the
origin is a node (benchmarks/sources.md). T7 is also the one pinned problem on
a box whose side is not 1 and whose lower corner is not the origin.
"""

import pytest

from maesolvers import T1, T2, T7, FDNewton, FDPoisson
from maesolvers.metrics import evaluate

PUBLISHED_T1 = {31: 7.14e-5, 63: 1.73e-5}
PUBLISHED_T2 = {31: 17.15e-3, 63: 12.53e-3}


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_T1.items()))
@pytest.mark.parametrize("solver", [FDNewton, FDPoisson])
def test_smooth_example_matches_froese_oberman(solver, n, expected):
    err = evaluate(solver(n=n).solve(T1))["max"]
    assert err == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_T2.items()))
def test_blowup_example_matches_froese_oberman_with_poisson_iteration(n, expected):
    err = evaluate(FDPoisson(n=n).solve(T2))["max"]
    assert err == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("solver", [FDNewton, FDPoisson])
def test_t7_minimum_matches_benamou_froese_oberman(solver):
    assert evaluate(solver(n=20).solve(T7))["u_origin"] == pytest.approx(
        0.2892, abs=5e-5
    )
