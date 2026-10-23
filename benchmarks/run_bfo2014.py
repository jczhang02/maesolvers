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

Every solve uses the preset FDTransport, or, where its mass-balance check
cannot run (``volume`` and ``integrate`` do not know the sets of Tables 2 and
3), the same scheme, start and iteration composed by hand
(:func:`unchecked`); the balance holds there by construction. Variants beside
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

Every solve is one MLflow run in maesolvers-bfo2014-<problem> (T16, T12,
ellipse, halfdisks) unless --no-mlflow is given; the solves run in
--processes worker processes, one thread each. Run it on a100server from the
repository root::

    just remote-bg bfo2014 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        MKL_NUM_THREADS=1 python benchmarks/run_bfo2014.py --processes 16
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from maesolvers import (
    T12,
    T16,
    Box,
    Disk,
    Ellipse,
    FDTransport,
    Newton,
    Problem,
    Solver,
    Transport,
    tracking,
)
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


def unchecked(preset: FDTransport) -> Newton:
    """``preset`` as a composed solver: its scheme, start and iteration, no balance check."""
    return Newton(
        scheme=preset.scheme,
        start=preset.start,
        max_iter=preset.max_iter,
        tol=preset.tol,
    )


def solver_for(name: str, variant: str, n: int, directions: int | None) -> Solver:
    """The solver of one row."""
    box = Box((-0.8, -0.8), (0.8, 0.8)) if variant == "tight_square" else SQUARES[name]
    preset = FDTransport(n=n, directions=directions, box=box)
    if variant == "poisson_start":
        return Newton(scheme=preset.scheme, max_iter=preset.max_iter, tol=preset.tol)
    if variant == "monotone":
        assert preset.start is not None
        return preset.start
    return preset if name in ("T16", "T12") else unchecked(preset)


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
    start = time.time()
    if log:
        import mlflow

        logger = tracking.MLflowLogger()
        with tracking.run(f"maesolvers-bfo2014-{name}", run_name=run_name):
            result = solver.solve(problem, callbacks=[logger])
            errors = gradient_errors(name, result)
            mlflow.log_metrics(errors)
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
        "converged": info["converged"],
        "residual_max": info["residual_max"],
        "u_anchor": info["u_anchor"],
        "monotone_fraction": info.get("monotone_fraction", ""),
        "published_time": pub[3],
        "time": info["time"],
        "wall": wall,
    }


def main() -> None:
    """Parse the command line, run the solves, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs")
    parser.add_argument("--processes", type=int, default=16, help="worker processes")
    parser.add_argument("--grids", type=int, nargs="+", default=list(GRIDS))
    parser.add_argument(
        "--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS)
    )
    parser.add_argument("--out", type=Path, default=HERE / "bfo2014.csv")
    args = parser.parse_args()
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
