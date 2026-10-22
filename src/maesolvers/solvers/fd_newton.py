"""Standard finite differences with a damped Newton iteration.

Centred second differences on a uniform grid over a :class:`Box`, Dirichlet
boundary values imposed at the boundary nodes, Newton's method on the
discrete residual ``u_xx u_yy - u_xy^2 - f`` with backtracking, started from
the Poisson problem ``laplace u = 2 sqrt(f)`` (Loeper-Rapetti 2005; the
standard discretisation of Benamou-Froese-Oberman 2010).

The scheme is not monotone: on smooth problems it converges to the convex
solution from the Poisson start, but it carries no convexity guarantee.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

from maesolvers.domains import Box
from maesolvers.problems import Problem
from maesolvers.result import Result


@dataclass
class FDNewton:
    """``n`` cells per axis, so ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns."""

    n: int = 64
    max_iter: int = 50
    tol: float = 1e-10
    """Stop when ``max |residual| <= tol * max(1, max f)``."""
    interpolation: str = "cubic"

    def solve(self, problem: Problem) -> Result:
        if problem.boundary != "dirichlet":
            raise ValueError("FDNewton handles Dirichlet boundary conditions only")
        if not isinstance(problem.domain, Box):
            raise TypeError(
                "FDNewton needs a Box domain; curved boundaries need a "
                "Shortley-Weller treatment that is not implemented yet"
            )
        t0 = time.perf_counter()
        box, n = problem.domain, self.n
        x = np.linspace(box.lo[0], box.hi[0], n + 1)
        y = np.linspace(box.lo[1], box.hi[1], n + 1)
        hx, hy = x[1] - x[0], y[1] - y[0]
        nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
        m = n + 1
        interior = np.zeros((m, m), dtype=bool)
        interior[1:-1, 1:-1] = True
        idx_int = np.flatnonzero(interior.ravel())
        idx_bnd = np.flatnonzero(~interior.ravel())

        dxx, dyy, dxy = _difference_operators(m, hx, hy, idx_int)
        f_int = problem.f(nodes.reshape(-1, 2)[idx_int])
        u = np.zeros(m * m)
        u[idx_bnd] = problem.g(nodes.reshape(-1, 2)[idx_bnd])

        # Columns acting on the unknowns and on the known boundary values.
        ops = [(d[:, idx_int].tocsc(), d[:, idx_bnd].tocsc()) for d in (dxx, dyy, dxy)]
        (axx, bxx), (ayy, byy), (axy, bxy) = ops
        ub = u[idx_bnd]

        # Poisson start: laplace u = 2 sqrt(f), the convex branch's trace.
        lap = (axx + ayy).tocsc()
        rhs = 2.0 * np.sqrt(f_int) - (bxx + byy) @ ub
        v = spsolve(lap, rhs)

        def residual(
            v: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            uxx = axx @ v + bxx @ ub
            uyy = ayy @ v + byy @ ub
            uxy = axy @ v + bxy @ ub
            return uxx * uyy - uxy**2 - f_int, uxx, uyy, uxy

        history = []
        r, uxx, uyy, uxy = residual(v)
        norm = np.max(np.abs(r))
        history.append(norm)
        stop = self.tol * max(1.0, float(np.max(np.abs(f_int))))
        it = 0
        stalled = False
        while norm > stop and it < self.max_iter and not stalled:
            jac = (
                sp.diags(uyy) @ axx + sp.diags(uxx) @ ayy - 2.0 * sp.diags(uxy) @ axy
            ).tocsc()
            step = spsolve(jac, -r)
            t = 1.0
            for _ in range(40):
                r_new, uxx_n, uyy_n, uxy_n = residual(v + t * step)
                norm_new = np.max(np.abs(r_new))
                if norm_new < (1.0 - 1e-4 * t) * norm:
                    break
                t *= 0.5
            else:
                # No step length decreases the residual: rounding level.
                stalled = True
                continue
            v = v + t * step
            r, uxx, uyy, uxy = r_new, uxx_n, uyy_n, uxy_n
            norm = norm_new
            history.append(norm)
            it += 1

        u[idx_int] = v
        grid = u.reshape(m, m)
        hess = np.full((m * m, 2, 2), np.nan)
        hess[idx_int, 0, 0] = uxx
        hess[idx_int, 1, 1] = uyy
        hess[idx_int, 0, 1] = hess[idx_int, 1, 0] = uxy
        interp = RegularGridInterpolator((x, y), grid, method=self.interpolation)
        return Result(
            problem=problem,
            u=lambda p: interp(np.asarray(p, dtype=float)),
            nodes=nodes.reshape(-1, 2),
            values=u,
            hessian=hess,
            info={
                "solver": "fd_newton",
                "n": n,
                "h": (hx, hy),
                "unknowns": idx_int.size,
                "iterations": it,
                "residual_history": history,
                "converged": bool(norm <= stop),
                "stalled": stalled,
                "time": time.perf_counter() - t0,
            },
        )


def _difference_operators(
    m: int, hx: float, hy: float, idx_int: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Centred ``u_xx``, ``u_yy``, ``u_xy`` from all nodes to interior nodes.

    Node ``(i, j)`` has flat index ``i * m + j``; ``i`` runs along x.
    """
    rows = np.arange(idx_int.size)

    def build(offsets: list[tuple[int, int, float]], scale: float) -> sp.csr_matrix:
        r, c, v = [], [], []
        for di, dj, w in offsets:
            r.append(rows)
            c.append(idx_int + di * m + dj)
            v.append(np.full(rows.size, w * scale))
        return sp.csr_matrix(
            (np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
            shape=(idx_int.size, m * m),
        )

    dxx = build([(1, 0, 1.0), (0, 0, -2.0), (-1, 0, 1.0)], 1.0 / hx**2)
    dyy = build([(0, 1, 1.0), (0, 0, -2.0), (0, -1, 1.0)], 1.0 / hy**2)
    dxy = build(
        [(1, 1, 1.0), (1, -1, -1.0), (-1, 1, -1.0), (-1, -1, 1.0)],
        1.0 / (4.0 * hx * hy),
    )
    return dxx, dyy, dxy
