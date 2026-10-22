"""Published numbers the solvers must match.

Source: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column
"Standard" (same values in Froese-Oberman, SIAM J. Numer. Anal. 49 (2011),
Table 3, column "Standard F.D."). Max error at the grid nodes on an N x N
grid with h = 1 / N; see benchmarks/standard_fd.md for the convention.
"""

import pytest

from maesolvers import P4, P5, FDNewton, FDPoisson
from maesolvers.metrics import evaluate

PUBLISHED_P4 = {31: 7.14e-5, 63: 1.73e-5}
PUBLISHED_P5 = {31: 17.15e-3, 63: 12.53e-3}


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_P4.items()))
@pytest.mark.parametrize("solver", [FDNewton, FDPoisson])
def test_smooth_example_matches_froese_oberman(solver, n, expected):
    err = evaluate(solver(n=n).solve(P4))["max_nodes"]
    assert err == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("n,expected", sorted(PUBLISHED_P5.items()))
def test_blowup_example_matches_froese_oberman_with_poisson_iteration(n, expected):
    err = evaluate(FDPoisson(n=n).solve(P5))["max_nodes"]
    assert err == pytest.approx(expected, rel=5e-3)
