"""Regenerate bfo2014.csv: the transport scheme on Tables 1 to 3 of BFO 2014.

Benamou, Froese and Oberman, J. Comput. Phys. 260 (2014): Table 1 maps the
square onto itself (T16, Section 6.1), Table 2 an ellipse onto a rotated
ellipse (Section 6.2), Table 3 two half-disks onto a disk (Section 6.3); each
prints the max over the source domain of ``|grad u - grad u_exact|`` (Table 1
also the L2 norm) against the grid size ``N_X`` and, for the curved targets,
the number ``N_Y`` of directions in the boundary equation. The two problems of
Tables 2 and 3 have no T id in the shared specification (Table 2 is a T13
variant with other matrices), so they are built here, with their target and
source sets as small Domain classes of this script. T12 runs too, at the same
grids, for the method page; it has no published table.

Every solve uses the preset FDTransport. It cannot check the mass balance
of Tables 2 and 3 (``volume`` and ``integrate`` do not know their sets) and
warns; the balance holds there by construction, so this script silences
that warning. Variants beside
the preset: the paper's Poisson start without the monotone root
(``poisson_start``), the monotone interior alone (``monotone``), ``h = 1/(N -
1)`` (``n = N - 1``) and, for Table 2, the smallest square around the ellipse.

The discrete gradient is the paper's centred difference (Section 3.1) at the
grid nodes of the source domain off the boundary ring, through
``gradient_fd(result.u, points, step=h)``: the spline through the node values
interpolates them, so these are the node differences to rounding. For T16,
``gradient_max_ring`` and ``gradient_rms_ring`` add the ring nodes, with
one-sided differences there. With --problems only those problems run, and
their rows replace the same problems' rows of the CSV given by --out.

With --checks it writes bfo2014_checks.csv (or --out) instead,
the evidence for Findings 2 and 3 of bfo2014.md, at N = 32, 64 and 128 with
N_Y = 512: the preset on the ellipse and T12 (``preset``); filter widths 3
and 1e6, the latter the standard 9-point scheme alone, on the ellipse
(``width``); the paper's full Newton step (``undamped``, run_filtered's
UndampedNewton, 300 steps) on the filtered scheme from the Poisson start
and from the monotone root, on T12 and the ellipse; and ``u(x_0)`` added to
``f`` inside the ellipse only (``anchor_inside``, filter width 1e6). Besides
the errors (max and median) and the iteration counts, each row holds the
gauge of the domain (1 on its boundary) at the node of the largest gradient
error, and statistics of the last iterate on the whole square: the smallest
eigenvalue of the centred Hessian at the interior nodes outside the domain
and the share of those nodes where it is negative, the same share inside,
the gauge at the node of the largest residual and the filter's argument
there, and how far the one-sided gradient on the square's boundary ring
lies off the target's boundary.

Every solve is one MLflow run in maesolvers-bfo2014-<problem> (T16, T12,
ellipse, halfdisks) unless --no-mlflow is given; the solves run in
--processes worker processes, one thread each. Run it on a100server from the
repository root::

    just remote-bg bfo2014 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        MKL_NUM_THREADS=1 python benchmarks/run_bfo2014.py --processes 16
    just remote-bg bfo2014-checks env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        MKL_NUM_THREADS=1 python benchmarks/run_bfo2014.py --checks \\
        --out /jczhang02/maesolvers-data/data/bfo2014_checks.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from run_filtered import UndampedNewton

from maesolvers import (
    T12,
    T16,
    Box,
    Disk,
    Ellipse,
    FDTransport,
    Newton,
    Problem,
    SecondBoundary,
    Solver,
    Transport,
    tracking,
)
from maesolvers.classical.schemes import monge_ampere
from maesolvers.classical.second_boundary import SecondBoundaryGrid
from maesolvers.core.domains import Domain
from maesolvers.metrics import evaluate, gradient_fd, gradient_map_error

HERE = Path(__file__).parent
GRIDS = (32, 64, 128, 256, 512)
"""N_X of Tables 1 to 3."""
DIRECTIONS = (32, 64, 128, 256, 512)
"""N_Y of Tables 2 and 3."""

# Published, as printed (Tables 1 to 3, checked against the page images).
TABLE1 = {  # N: (max error, L2 error, iterations, CPU seconds)
    32: ("0.0220", "0.0127", 2, 0.05),
    64: ("0.0110", "0.0064", 2, 0.18),
    128: ("0.0055", "0.0032", 3, 1.16),
    256: ("0.0028", "0.0016", 3, 6.44),
    512: ("0.0014", "0.0008", 3, 32.86),
}
TABLE2 = {  # N: (max error at N_Y = 32, ..., 512), iterations, seconds at N_Y = 512
    32: (("0.0691", "0.0667", "0.0662", "0.0660", "0.0660"), 4, 0.2),
    64: (("0.0306", "0.0284", "0.0279", "0.0277", "0.0277"), 4, 0.5),
    128: (("0.0203", "0.0176", "0.0169", "0.0167", "0.0167"), 4, 1.7),
    256: (("0.0127", "0.0096", "0.0088", "0.0086", "0.0088"), 5, 10.1),
    512: (("0.0086", "0.0056", "0.0047", "0.0045", "0.0047"), 5, 52.2),
}
TABLE3 = {
    32: (("0.0280", "0.0284", "0.0286", "0.0286", "0.0286"), 3, 0.2),
    64: (("0.0158", "0.0164", "0.0165", "0.0165", "0.0165"), 3, 0.4),
    128: (("0.0092", "0.0093", "0.0092", "0.0092", "0.0092"), 3, 1.3),
    256: (("0.0047", "0.0036", "0.0036", "0.0036", "0.0036"), 4, 8.3),
    512: (("0.0049", "0.0040", "0.0034", "0.0033", "0.0033"), 5, 51.7),
}

# Section 6.2: X = M_X B_1, Y = M_Y B_1, grad u = M_Y R_theta M_X^-1 x with
# tan(theta) = trace(M_X^-1 M_Y^-1 J) / trace(M_X^-1 M_Y^-1), J = R_(pi/2).
M_X = np.diag([0.8, 0.4])
M_Y = np.array([[0.6, 0.2], [0.2, 0.8]])


def ellipse_map() -> np.ndarray:
    """The matrix of the optimal map of Section 6.2, symmetric positive definite."""
    inverse = np.linalg.inv(M_X) @ np.linalg.inv(M_Y)
    turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    theta = np.arctan2(np.trace(inverse @ turn), np.trace(inverse))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return M_Y @ rotation @ np.linalg.inv(M_X)


@dataclass(frozen=True)
class MatrixEllipse:
    """The ellipse ``M B_1``, ``M`` symmetric positive definite, as a target.

    A :class:`~maesolvers.core.domains.Domain` with the method of
    :class:`~maesolvers.core.domains.SupportFunction`: ``H*(n) = |M n|``.
    """

    matrix: tuple[tuple[float, float], tuple[float, float]]

    @property
    def dim(self) -> int:
        """Always 2."""
        return 2

    @property
    def bbox(self) -> Box:
        """The box ``-/+ (|row_1 of M|, |row_2 of M|)``."""
        r = np.linalg.norm(np.asarray(self.matrix), axis=1)
        return Box(tuple(-r), tuple(r))

    def _level(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=float) @ np.linalg.inv(np.asarray(self.matrix)).T
        return np.sum(z**2, axis=-1)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closed ellipse."""
        return self._level(x) <= 1.0

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the open ellipse."""
        return self._level(x) < 1.0

    def support_function(self, n: np.ndarray) -> np.ndarray:
        """``|M n|`` for each direction."""
        return np.linalg.norm(
            np.asarray(n, dtype=float) @ np.asarray(self.matrix).T, axis=-1
        )


@dataclass(frozen=True)
class HalfDisks:
    """The source of Section 6.3, two half-disks of radius 0.85.

    ``x1 < -0.2`` and ``|x + (0.2, 0)| < 0.85``, or ``x1 > 0.1`` and
    ``|x - (0.1, 0)| < 0.85``.
    """

    @property
    def dim(self) -> int:
        """Always 2."""
        return 2

    @property
    def bbox(self) -> Box:
        """``[-1.05, 0.95] x [-0.85, 0.85]``."""
        return Box((-1.05, -0.85), (0.95, 0.85))

    def _inside(self, x: np.ndarray, closed: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        less = np.less_equal if closed else np.less
        out = np.zeros(x.shape[:-1], dtype=bool)
        for centre, side in ((-0.2, -1.0), (0.1, 1.0)):
            offset = x - np.array([centre, 0.0])
            radius = less(np.sum(offset**2, axis=-1), 0.85**2)
            out |= less(0.0, side * offset[..., 0]) & radius
        return out

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in the closure of either half-disk."""
        return self._inside(x, closed=True)

    def interior(self, x: np.ndarray) -> np.ndarray:
        """Whether each point lies in either open half-disk."""
        return self._inside(x, closed=False)


def _constant(value: float):
    def f(x: np.ndarray) -> np.ndarray:
        return np.full(np.shape(x)[:-1], value)

    return f


ELLIPSE = Problem(
    "ellipse",
    Ellipse((0.0, 0.0), (0.8, 0.4)),
    # Uniform densities on X and Y: f = |Y| / |X| = det M_Y / det M_X.
    _constant(float(np.linalg.det(M_Y) / np.linalg.det(M_X))),
    Transport(MatrixEllipse(tuple(map(tuple, M_Y)))),  # type: ignore[arg-type]
    exact=lambda x: 0.5 * np.einsum("...i,ij,...j->...", x, ellipse_map(), x),
    note="BFO 2014, Section 6.2: the ellipse M_X B_1 onto M_Y B_1.",
)
HALF_DISKS = Problem(
    "halfdisks",
    HalfDisks(),  # type: ignore[arg-type]
    _constant(1.0),  # the two half-disks have the area of the target disk
    Transport(Disk((0.0, 0.0), 0.85)),
    note="BFO 2014, Section 6.3: two half-disks onto the disk of radius 0.85.",
)
PROBLEMS = {"T16": T16, "T12": T12, "ellipse": ELLIPSE, "halfdisks": HALF_DISKS}
SQUARES = {  # the square of the grid; None is the domain's bounding box
    "T16": None,
    "T12": None,
    "ellipse": Box((-1.0, -1.0), (1.0, 1.0)),
    "halfdisks": Box((-1.05, -1.0), (0.95, 1.0)),
}


def exact_gradient(name: str, x: np.ndarray) -> np.ndarray:
    """The exact map at points of the source domain."""
    if name == "ellipse":
        return x @ ellipse_map().T
    if name == "halfdisks":
        shift = np.where(x[..., :1] < 0.0, 0.2, -0.1)
        return x + np.concatenate([shift, np.zeros_like(shift)], axis=-1)
    exact = PROBLEMS[name].exact
    assert exact is not None
    return gradient_fd(exact, x)


def solver_for(name: str, variant: str, n: int, directions: int | None) -> Solver:
    """The solver of one row."""
    box = Box((-0.8, -0.8), (0.8, 0.8)) if variant == "tight_square" else SQUARES[name]
    preset = FDTransport(n=n, directions=directions, box=box)
    if variant == "poisson_start":
        return Newton(scheme=preset.scheme, max_iter=preset.max_iter, tol=preset.tol)
    if variant == "monotone":
        assert preset.start is not None
        return preset.start
    return preset


def gradient_errors(name: str, result: Any) -> dict[str, float]:
    """Max and RMS of the gradient-map error at the interior nodes of the domain."""
    h = result.info["h"][0]
    points = result.nodes[result.interior]
    out = gradient_map_error(
        gradient_fd(result.u, points, step=h), exact_gradient(name, points)
    )
    if name == "T16":  # every node, one-sided differences on the ring
        n = round(1.0 / h)
        values = result.values.reshape(n + 1, n + 1)
        gradient = np.stack(np.gradient(values, h, edge_order=1), axis=-1)
        exact = exact_gradient(name, result.nodes).reshape(n + 1, n + 1, 2)
        ring = gradient_map_error(gradient, exact)
        out["gradient_max_ring"] = ring["gradient_max"]
        out["gradient_rms_ring"] = ring["gradient_rms"]
    return out


def published(table: int, N: int, ny_index: int | None) -> tuple[str, str, Any, Any]:
    """Published max, L2, iterations and time of one cell, "" where none."""
    if table == 1:
        mx, l2, iterations, seconds = TABLE1[N]
        return mx, l2, iterations, seconds
    values, iterations, seconds = (TABLE2 if table == 2 else TABLE3)[N]
    assert ny_index is not None
    last = ny_index == len(DIRECTIONS) - 1  # iterations and time are for N_Y = 512
    return values[ny_index], "", iterations if last else "", seconds if last else ""


TABLES = {"T16": 1, "ellipse": 2, "halfdisks": 3, "T12": 0}


def tasks() -> list[tuple[str, str, int, int | None]]:
    """(problem, variant, N, N_Y) of every row, the largest grids first."""
    out: list[tuple[str, str, int, int | None]] = []
    for N in GRIDS:
        for variant in ("preset", "poisson_start", "monotone", "h=1/(N-1)"):
            out.append(("T16", variant, N, None))
        out.append(("T12", "preset", N, None))
        for name in ("ellipse", "halfdisks"):
            out.extend((name, "preset", N, ny) for ny in DIRECTIONS)
            for variant in ("poisson_start", "monotone", "h=1/(N-1)"):
                out.append((name, variant, N, 512))
        out.append(("ellipse", "tight_square", N, 512))
    out.sort(key=lambda t: -t[2])
    return out


def run(task: tuple[str, str, int, int | None, bool]) -> dict[str, Any]:
    """One row of bfo2014.csv."""
    name, variant, N, ny, log = task
    n = N - 1 if variant == "h=1/(N-1)" else N
    problem = PROBLEMS[name]
    solver = solver_for(name, variant, n, ny)
    run_name = f"N={N} NY={ny} {variant}"
    warnings.filterwarnings("ignore", "FDTransport does not check the mass balance")
    start = time.time()
    if log:
        import mlflow

        logger = tracking.MLflowLogger()
        with tracking.run(f"maesolvers-bfo2014-{name}", run_name=run_name):
            result = solver.solve(problem, callbacks=[logger])
            errors = gradient_errors(name, result)
            begun = {
                k: float(result.info[k])
                for k in ("start_iterations", "start_converged")
                if k in result.info
            }
            mlflow.log_metrics({**errors, **begun})
    else:
        result = solver.solve(problem)
        errors = gradient_errors(name, result)
    wall = time.time() - start
    table = TABLES[name]
    ny_index = DIRECTIONS.index(ny) if ny in DIRECTIONS else None
    pub = published(table, N, ny_index) if table else ("", "", "", "")
    measured = evaluate(result) if problem.exact is not None else {}
    info = result.info
    return {
        "table": table or "",
        "problem": name,
        "variant": variant,
        "N": N,
        "n": n,
        "NY": "" if ny is None else ny,
        "published_max": pub[0],
        "gradient_max": errors["gradient_max"],
        "published_l2": pub[1],
        "gradient_rms": errors["gradient_rms"],
        "gradient_max_ring": errors.get("gradient_max_ring", ""),
        "gradient_rms_ring": errors.get("gradient_rms_ring", ""),
        "u_max": measured.get("max", ""),
        "u_rms": measured.get("rms", ""),
        "published_iterations": pub[2],
        "iterations": info["iterations"],
        "start_iterations": info.get("start_iterations", ""),
        "start_converged": info.get("start_converged", ""),
        "converged": info["converged"],
        "residual_max": info["residual_max"],
        "u_anchor": info["u_anchor"],
        "monotone_fraction": info.get("monotone_fraction", ""),
        "published_time": pub[3],
        "time": info["time"],
        "wall": wall,
    }


CHECK_GRIDS = (32, 64, 128)
"""N_X of the checks."""
UNDAMPED_STEPS = 300
"""Full Newton steps of the ``undamped`` check."""


@dataclass(frozen=True, kw_only=True)
class AnchorInside(SecondBoundary):
    """SecondBoundary with ``u(x_0)`` added to ``f`` inside the source domain only.

    SecondBoundary, as the paper, adds it at every interior node of the
    square; this variant leaves the equation ``MA[u] = 0`` at the nodes
    outside the closed domain.
    """

    def residual(self, grid: SecondBoundaryGrid, v: np.ndarray) -> np.ndarray:
        """SecondBoundary's residual without ``- u(x_0)`` outside the domain."""
        out = super().residual(grid, v)
        out[_outside(grid)] += v[grid.anchor]
        return out

    def jacobian(self, grid: SecondBoundaryGrid, v: np.ndarray) -> sp.csc_matrix:
        """SecondBoundary's Jacobian without the ``-1`` of ``x_0`` outside."""
        rows = np.flatnonzero(_outside(grid))
        undo = sp.csc_matrix(
            (np.ones(len(rows)), (rows, np.full(len(rows), grid.anchor))),
            shape=(len(v), len(v)),
        )
        return sp.csc_matrix(super().jacobian(grid, v) + undo)


def _outside(grid: SecondBoundaryGrid) -> np.ndarray:
    """The interior nodes of the square outside the closed source domain."""
    return grid.interior & ~grid.inside


def gauge(domain: Domain, x: np.ndarray) -> np.ndarray:
    """``|A^-1 (x - c)|`` for an ellipse ``c + A B_1``: 1 on its boundary."""
    if isinstance(domain, MatrixEllipse):
        return np.linalg.norm(x @ np.linalg.inv(np.asarray(domain.matrix)).T, axis=-1)
    if isinstance(domain, Disk):
        return np.linalg.norm(x - np.asarray(domain.center), axis=-1) / domain.radius
    assert isinstance(domain, Ellipse)
    offset = (x - np.asarray(domain.center)) / np.asarray(domain.semi_axes)
    return np.linalg.norm(offset, axis=-1)


def check_solver(check: str, name: str, N: int, variant: Any) -> Newton:
    """The solver of one row of bfo2014_checks.csv, ``n = N`` cells."""
    box = SQUARES[name]
    if check == "preset":
        return FDTransport(n=N, box=box)
    if check == "width":
        return FDTransport(n=N, epsilon=variant, box=box)
    if check == "undamped":
        start = (
            Newton(
                scheme=SecondBoundary(n=N, interior="monotone", box=box), max_iter=200
            )
            if variant == "monotone"
            else None
        )
        scheme = SecondBoundary(n=N, box=box)
        return UndampedNewton(scheme=scheme, start=start, max_iter=UNDAMPED_STEPS)
    assert check == "anchor_inside"
    monotone = AnchorInside(n=N, interior="monotone", box=box)
    scheme = AnchorInside(n=N, epsilon=variant, box=box)
    return Newton(
        scheme=scheme, start=Newton(scheme=monotone, max_iter=200), max_iter=200
    )


def iterate_statistics(
    problem: Problem, scheme: SecondBoundary, v: np.ndarray
) -> dict[str, float]:
    """Statistics of the unknowns ``v`` on the whole square, for the findings."""
    assert isinstance(problem.boundary, Transport)
    grid = scheme.discretize(problem)
    box_grid, interior = grid.box_grid, grid.interior
    uxx, uyy, uxy = grid.dxx @ v, grid.dyy @ v, grid.dxy @ v
    smallest = 0.5 * (uxx + uyy) - np.hypot(0.5 * (uxx - uyy), uxy)
    outside = ~grid.inside[interior]
    residual = np.abs(scheme.residual(grid, v))
    worst = int(np.argmax(residual))
    argument = float("nan")  # on the ring, or no filter
    if interior[worst] and grid.width is not None:
        at = int(np.sum(interior[:worst]))  # its row among the interior nodes
        monotone = scheme.interior_operator(dataclasses.replace(grid, width=None), v)
        accurate = monge_ampere(uxx, uyy, uxy)
        argument = float((accurate[at] - monotone[at]) / grid.width)
    n = box_grid.n
    h = box_grid.h[0]
    values = v.reshape(n + 1, n + 1)
    gradient = np.stack(np.gradient(values, h, edge_order=1), axis=-1).reshape(-1, 2)
    ring = gauge(problem.boundary.target, gradient[~interior])
    return {
        "min_eigenvalue_outside": float(np.min(smallest[outside])),
        "nonconvex_outside": float(np.mean(smallest[outside] < 0.0)),
        "nonconvex_inside": float(np.mean(smallest[~outside] < 0.0)),
        "worst_residual_gauge": float(gauge(problem.domain, box_grid.nodes[worst])),
        "worst_residual_on_ring": float(not interior[worst]),
        "worst_filter_argument": argument,
        "ring_offset": float(np.max(np.abs(ring - 1.0))),
    }


def check_measures(name: str, solver: Newton, result: Any) -> dict[str, float]:
    """The errors, the iteration counts and :func:`iterate_statistics` of a solve."""
    info = result.info
    points = result.nodes[result.interior]
    error = np.linalg.norm(
        gradient_fd(result.u, points, step=info["h"][0]) - exact_gradient(name, points),
        axis=-1,
    )
    scheme = solver.scheme
    assert isinstance(scheme, SecondBoundary)
    problem = PROBLEMS[name]
    v = result.u(scheme.discretize(problem).box_grid.nodes)  # the spline interpolates
    nan = float("nan")
    return {
        "gradient_max": float(np.max(error)),
        "gradient_median": float(np.median(error)),
        "worst_error_gauge": float(gauge(problem.domain, points[np.argmax(error)])),
        "iterations": info["iterations"],
        "start_iterations": info.get("start_iterations", nan),
        "start_converged": float(info.get("start_converged", nan)),
        "converged": float(info["converged"]),
        "residual_max": info["residual_max"],
        "residual_min": float(np.min(info["history"], initial=info["residual_max"])),
        "u_anchor": info["u_anchor"],
        "monotone_fraction": info.get("monotone_fraction", nan),
        **iterate_statistics(problem, scheme, v),
        "time": info["time"],
    }


def checks(task: tuple[str, str, int, Any, bool]) -> dict[str, Any]:
    """One row of bfo2014_checks.csv.

    ``task`` is (check, problem, N, variant, log): the variant is the filter
    width for "width" and "anchor_inside", the start ("poisson" or
    "monotone") for "undamped", unused for "preset". Booleans are written
    as 0 or 1, and a value that does not apply as an empty cell.
    """
    check, name, N, variant, log = task
    solver = check_solver(check, name, N, variant)
    warnings.filterwarnings("ignore", "FDTransport does not check the mass balance")
    run_name = f"check={check} N={N} variant={variant}"
    experiment = f"maesolvers-bfo2014-{name}"
    with tracking.run(experiment, run_name=run_name) if log else nullcontext():
        callbacks = [tracking.MLflowLogger()] if log else []
        result = solver.solve(PROBLEMS[name], callbacks=callbacks)
        measured = check_measures(name, solver, result)
        if log:
            import mlflow

            mlflow.log_metrics({k: v for k, v in measured.items() if np.isfinite(v)})
    print(f"{name} {run_name}: max {measured['gradient_max']:.4f}", flush=True)
    cells = {k: "" if np.isnan(v) else v for k, v in measured.items()}
    return {"check": check, "problem": name, "N": N, "variant": variant, **cells}


CHECK_TASKS: list[tuple[str, str, int, Any]] = [
    *(("preset", name, N, "") for name in ("ellipse", "T12") for N in CHECK_GRIDS),
    *(("width", "ellipse", N, eps) for N in CHECK_GRIDS for eps in (3.0, 1e6)),
    *(
        ("undamped", name, N, start)
        for name in ("ellipse", "T12")
        for N in CHECK_GRIDS
        for start in ("poisson", "monotone")
    ),
    *(("anchor_inside", "ellipse", N, 1e6) for N in CHECK_GRIDS),
]
"""(check, problem, N, variant) of every row of bfo2014_checks.csv."""


def main() -> None:
    """Parse the command line, run the solves, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=16, help="worker processes")
    parser.add_argument("--grids", type=int, nargs="+", default=list(GRIDS))
    parser.add_argument(
        "--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS)
    )
    parser.add_argument(
        "--checks", action="store_true", help="write bfo2014_checks.csv only"
    )
    parser.add_argument(
        "--out", type=Path, help="the CSV (bfo2014.csv or bfo2014_checks.csv here)"
    )
    args = parser.parse_args()
    if args.checks:
        todo_checks = [(*task, not args.no_mlflow) for task in CHECK_TASKS]
        with Pool(args.processes) as pool:
            found = pool.map(checks, todo_checks, 1)
        out = args.out or HERE / "bfo2014_checks.csv"
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(found[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(found)
        print(f"wrote {len(found)} rows to {out}")
        return
    args.out = args.out or HERE / "bfo2014.csv"
    if tuple(args.grids) != GRIDS and args.out == HERE / "bfo2014.csv":
        parser.error("--grids differs from the tables': give --out")
    todo = [
        (*t, not args.no_mlflow)
        for t in tasks()
        if t[2] in args.grids and t[0] in args.problems
    ]
    with Pool(args.processes) as pool:
        rows = pool.map(run, todo, 1)
    fields = list(rows[0])  # a fresh row has every column, older ones may not
    if args.out.exists() and set(args.problems) != set(PROBLEMS):
        with args.out.open(newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if r["problem"] not in args.problems]
    order = {
        (t[0], t[1], t[2], t[3]): i for i, t in enumerate(sorted(tasks(), key=_key))
    }
    rows.sort(
        key=lambda r: order[
            (r["problem"], r["variant"], int(r["N"]), int(r["NY"]) if r["NY"] else None)
        ]
    )
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


def _key(task: tuple[str, str, int, int | None]) -> tuple[Any, ...]:
    """Table order: problem, variant, N, N_Y."""
    name, variant, N, ny = task
    variants = ("preset", "poisson_start", "monotone", "h=1/(N-1)", "tight_square")
    problems = ("T16", "ellipse", "halfdisks", "T12")
    return problems.index(name), variants.index(variant), N, ny or 0


if __name__ == "__main__":
    main()
