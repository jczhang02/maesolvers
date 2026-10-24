"""Sparse M-matrix solves, for Howard's algorithm and the semi-Lagrangian scheme.

:class:`~maesolvers.classical.iterations.Howard` and
:meth:`~maesolvers.classical.semilagrangian.SemiLagrangian.solve_laplace` use
:func:`solve_m_matrix`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, gmres, spsolve

_RTOL = 1e-10
"""Relative residual at which the Krylov methods stop, below any iteration's tolerance."""

_ACCEPT = 1e-8
"""Relative residual up to which a Krylov solution is kept, even after a breakdown."""

_RESTART = 50
"""Krylov vectors GMRES keeps between restarts."""


def solve_m_matrix(matrix: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve a sparse system whose matrix is, up to sign, an M-matrix.

    Both sides are divided by the diagonal, and the system is solved by
    BiCGSTAB to a relative residual of ``1e-10``; where BiCGSTAB stops above
    ``1e-8`` (it breaks down when the right-hand side is at the rounding
    level), by restarted GMRES from its last iterate; where that stops above
    ``1e-8`` too, by a sparse LU factorisation. For the M-matrices of a
    monotone wide-stencil scheme the Krylov methods need fewer steps the
    wider the stencil, where an LU factorisation fills in.

    Parameters
    ----------
    matrix
        Square and sparse, with no zero on the diagonal.
    rhs
        The right-hand side.

    Returns
    -------
    numpy.ndarray
        The solution.
    """
    matrix = sp.csr_matrix(matrix)
    scale = 1.0 / np.asarray(matrix.diagonal(), dtype=float)
    scaled, b = sp.csr_matrix(sp.diags(scale) @ matrix), scale * rhs
    target = _ACCEPT * np.linalg.norm(b)
    budget = 10 * len(b)
    x, _ = bicgstab(scaled, b, rtol=_RTOL, atol=0.0, maxiter=budget)
    if np.linalg.norm(scaled @ x - b) <= target:
        return x
    x, _ = gmres(
        scaled,
        b,
        x0=x if np.all(np.isfinite(x)) else None,
        rtol=_RTOL,
        atol=0.0,
        restart=_RESTART,
        maxiter=budget // _RESTART + 1,
    )
    if np.linalg.norm(scaled @ x - b) <= target:
        return x
    return np.asarray(spsolve(matrix.tocsc(), rhs))
