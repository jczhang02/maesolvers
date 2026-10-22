"""The Poisson iteration of Benamou, Froese and Oberman (2010), method (M2).

Same standard 9-point discretisation as :class:`FDNewton`, solved to its
fixed point by the iteration

    laplace u^{k+1} = sqrt((u^k_xx)^2 + (u^k_yy)^2 + 2 (u^k_xy)^2 + 2 f),

with Dirichlet data, started from ``laplace u^0 = sqrt(2 f)``.
The right-hand side is the Laplacian of a convex solution of the discrete
equation, so the iteration selects the convex branch where Newton's method
on the same discretisation may not (Froese-Oberman 2011, Section 8.1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import factorized

from maesolvers.domains import Box
from maesolvers.problems import Problem
from maesolvers.result import Result
from maesolvers.solvers.fd_newton import _difference_operators


@dataclass
class FDPoisson:
    """``n`` cells per axis; stop when successive iterates differ by ``tol``."""

    n: int = 64
    max_iter: int = 10_000
    tol: float = 1e-12
    interpolation: str = "cubic"

    def solve(self, problem: Problem) -> Result:
        if problem.boundary != "dirichlet":
            raise ValueError("FDPoisson handles Dirichlet boundary conditions only")
        if not isinstance(problem.domain, Box):
            raise TypeError("FDPoisson needs a Box domain")
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
        flat = nodes.reshape(-1, 2)
        f_int = problem.f(flat[idx_int])
        u = np.zeros(m * m)
        u[idx_bnd] = problem.g(flat[idx_bnd])
        ub = u[idx_bnd]
        ops = [(d[:, idx_int].tocsc(), d[:, idx_bnd].tocsc()) for d in (dxx, dyy, dxy)]
        (axx, bxx), (ayy, byy), (axy, bxy) = ops
        solve_lap = factorized((axx + ayy).tocsc())
        lap_b = (bxx + byy) @ ub

        v = solve_lap(np.sqrt(2.0 * f_int) - lap_b)
        history = []
        it = 0
        diff = np.inf
        while diff > self.tol and it < self.max_iter:
            uxx = axx @ v + bxx @ ub
            uyy = ayy @ v + byy @ ub
            uxy = axy @ v + bxy @ ub
            rhs = np.sqrt(uxx**2 + uyy**2 + 2.0 * uxy**2 + 2.0 * f_int)
            v_new = solve_lap(rhs - lap_b)
            diff = float(np.max(np.abs(v_new - v)))
            history.append(diff)
            v = v_new
            it += 1

        uxx = axx @ v + bxx @ ub
        uyy = ayy @ v + byy @ ub
        uxy = axy @ v + bxy @ ub
        u[idx_int] = v
        hess = np.full((m * m, 2, 2), np.nan)
        hess[idx_int, 0, 0] = uxx
        hess[idx_int, 1, 1] = uyy
        hess[idx_int, 0, 1] = hess[idx_int, 1, 0] = uxy
        interp = RegularGridInterpolator(
            (x, y), u.reshape(m, m), method=self.interpolation
        )
        return Result(
            problem=problem,
            u=lambda p: interp(np.asarray(p, dtype=float)),
            nodes=flat,
            values=u,
            hessian=hess,
            info={
                "solver": "fd_poisson",
                "n": n,
                "h": (hx, hy),
                "unknowns": idx_int.size,
                "iterations": it,
                "update_history": history,
                "converged": bool(diff <= self.tol),
                "residual_max": float(np.max(np.abs(uxx * uyy - uxy**2 - f_int))),
                "time": time.perf_counter() - t0,
            },
        )
