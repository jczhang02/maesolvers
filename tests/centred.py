"""The centred smooth example of Froese-Oberman (2013), shared by the tests.

Their eq. (4.1), ``u = exp(|x - x0|^2 / 2)`` with ``x0 = (1/2, 1/2)`` on the
unit square: T1 centred, which the catalogue lacks (``benchmarks/filtered.md``,
Finding 1). ``benchmarks/run_filtered.py`` builds the same problem.
"""

import numpy as np

from maesolvers import Box, Dirichlet, Problem

X0 = np.array([0.5, 0.5])


def _centred_exp(x):
    return np.exp(np.sum((np.asarray(x, dtype=float) - X0) ** 2, axis=-1) / 2)


def _centred_exp_f(x):
    r2 = np.sum((np.asarray(x, dtype=float) - X0) ** 2, axis=-1)
    return (1 + r2) * np.exp(r2)


C2_CENTRED = Problem(
    "C2-centred",
    Box((0.0, 0.0), (1.0, 1.0)),
    _centred_exp_f,
    Dirichlet(_centred_exp),
    exact=_centred_exp,
    note="FO 2013, eq. (4.1): exp(|x - x0|^2 / 2), x0 = (1/2, 1/2); T1 centred.",
)
