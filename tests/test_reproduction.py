"""Published numbers the solvers must match.

Source: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column
"Standard" (same values in Froese-Oberman, SIAM J. Numer. Anal. 49 (2011),
Table 3, column "Standard F.D."). Max error at the grid nodes on an N x N
grid with h = 1 / N; see benchmarks/standard_fd.md for the convention.

Benamou-Froese-Oberman, ESAIM: M2AN 44 (2010), Table 7, column "(M1), (M2)":
u_min of T7 on [-1, 1]^2, where N counts nodes per side, so N = 21 is n = 20
cells and the origin is a node (benchmarks/standard_fd.md, Finding 3).
FDPoisson matches it at every printed N (benchmarks/t7_bfo2010.csv); the first
three are pinned here. FDNewton does not (a non-convex root from N = 41), so
it is not pinned. T7 is also the one pinned problem on a box whose side is not
1 and whose lower corner is not the origin.
"""

import pytest

from maesolvers import T1, T2, T7, FDNewton, FDPoisson
from maesolvers.metrics import evaluate

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
