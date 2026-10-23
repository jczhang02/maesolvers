"""Check that a change leaves the 2D finite-difference results equal bit for bit.

``snapshot OUT.npz`` solves every finite-difference preset (FDNewton,
FDPoisson, FDMonotone with 9, 17 and 33 points and both boundary
treatments, FDHybrid, FDFiltered) on T1, T2, T20 and, for the standard
scheme, T8 and T6, at n = 8, 16 and 25, and FDNewton and FDMonotone on T1
at spline degrees 1, 2, 4 and 5. It stores, for each solve, the node
values, the node Hessian, the nodes and interior mask, the spline on the
shared evaluation grid, the fields of ``evaluate`` but ``time``, the final
residual and iteration count and the interpolant's name, and the sparse
matrices of ``difference_operators`` and ``directional_difference``.
``compare A.npz B.npz`` prints how many arrays differ, by shape, dtype or
``numpy.array_equal`` (NaN equal to NaN), and exits 1 if any does.

The script imports only names that maesolvers has had since 0.1, and calls
``directional_difference`` in its 0.1 signature ``(x, y, direction, g)`` or
its later one ``(grid, direction, g)``, so one copy snapshots any checkout
from 4fdd5da on. On a100server, from the repository
root of the branch under test (nd_grids.md)::

    git worktree add --detach /tmp/ref 4fdd5da
    PYTHONPATH=/tmp/ref/src python benchmarks/check_2d_bit_for_bit.py snapshot ref.npz
    python benchmarks/check_2d_bit_for_bit.py snapshot branch.npz
    python benchmarks/check_2d_bit_for_bit.py compare ref.npz branch.npz

Nothing is logged to MLflow: the check produces no numbers, only equality.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any

import numpy as np

from maesolvers import (
    T1,
    T2,
    T6,
    T8,
    T20,
    FDFiltered,
    FDHybrid,
    FDMonotone,
    FDNewton,
    FDPoisson,
)
from maesolvers.classical.monotone import directional_difference
from maesolvers.classical.schemes import difference_operators
from maesolvers.metrics import evaluate, shared_grid


def cases() -> list[tuple[str, Any, Any]]:
    """Every (name, solver, problem) the snapshot solves."""
    out: list[tuple[str, Any, Any]] = []
    for n in (8, 16, 25):
        for p in (T1, T2, T20, T8, T6):
            out += [
                (f"FDNewton-{p.name}-{n}", FDNewton(n=n), p),
                (f"FDPoisson-{p.name}-{n}", FDPoisson(n=n), p),
            ]
        for p in (T1, T2, T20):
            for points in (9, 17, 33):
                solver = FDMonotone(n=n, stencil=points)
                out.append((f"FDMonotone{points}-{p.name}-{n}", solver, p))
            solver = FDMonotone(n=n, boundary="interpolate")
            out.append((f"FDMonotoneI-{p.name}-{n}", solver, p))
            out.append((f"FDHybrid-{p.name}-{n}", FDHybrid(n=n), p))
            out.append((f"FDFiltered-{p.name}-{n}", FDFiltered(n=n), p))
        for degree in (1, 2, 4, 5):
            for preset in (FDNewton, FDMonotone):
                solver = preset(n=n, spline_degree=degree)
                out.append((f"{preset.__name__}-deg{degree}-T1-{n}", solver, T1))
    return out


def _directional_difference(
    x: np.ndarray, y: np.ndarray, direction: tuple[int, int], g: Any
) -> Any:
    """``directional_difference`` on the axes ``x`` and ``y``, in either signature."""
    if "grid" not in inspect.signature(directional_difference).parameters:
        return directional_difference(x, y, direction, g)  # 0.1
    from maesolvers import Box, BoxGrid

    box = Box((float(x[0]), float(y[0])), (float(x[-1]), float(y[-1])))
    return directional_difference(BoxGrid(box, len(x) - 1), direction, g)


def snapshot() -> dict[str, np.ndarray]:
    """Every array the check compares, by name."""
    out: dict[str, np.ndarray] = {}
    for name, solver, problem in cases():
        r = solver.solve(problem)
        out[name + "/values"] = r.values
        out[name + "/hess"] = r.node_hessian
        out[name + "/nodes"] = r.nodes
        out[name + "/interior"] = r.interior
        out[name + "/u"] = r.u(shared_grid(problem.domain))
        e = evaluate(r)
        out[name + "/eval"] = np.array([e[k] for k in sorted(e) if k != "time"])
        out[name + "/res"] = np.array([r.info["residual_max"], r.info["iterations"]])
        out[name + "/interp"] = np.array([str(r.info.get("interpolant"))])
    for n, hx, hy in ((4, 1.0, 1.0), (9, 0.1, 0.37)):
        for k, m in enumerate(difference_operators(n, hx, hy)):
            m = m.tocsr()
            out[f"do{n}-{k}/data"] = m.data
            out[f"do{n}-{k}/indices"] = m.indices
            out[f"do{n}-{k}/indptr"] = m.indptr
    x = np.linspace(-0.5, 0.9, 8)
    y = np.linspace(0.0, 1.4, 8)
    for direction in ((1, 0), (0, 1), (1, 1), (2, 1), (-1, 2), (3, -1)):
        a, c, full = _directional_difference(
            x, y, direction, lambda q: np.exp(q[..., 0]) * q[..., 1]
        )
        a = a.tocsr()
        out[f"dd{direction}/data"] = a.data
        out[f"dd{direction}/ind"] = a.indices
        out[f"dd{direction}/ptr"] = a.indptr
        out[f"dd{direction}/c"] = c
        out[f"dd{direction}/full"] = full
    return out


def differing(a: Any, b: Any) -> list[str]:
    """The names of the arrays of two snapshots that are not equal bit for bit."""
    if set(a.files) != set(b.files):
        raise SystemExit(
            f"the snapshots hold different arrays: {set(a.files) ^ set(b.files)}"
        )
    return [
        k
        for k in sorted(a.files)
        if not (
            a[k].shape == b[k].shape
            and a[k].dtype == b[k].dtype
            and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")
        )
    ]


def main() -> None:
    """Parse the command line and snapshot or compare."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot").add_argument("out")
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    args = parser.parse_args()
    if args.command == "snapshot":
        arrays = snapshot()
        np.savez(args.out, **arrays)
        print(len(cases()), "cases,", len(arrays), "arrays")
        return
    a, b = np.load(args.a), np.load(args.b)
    bad = differing(a, b)
    print(len(a.files), "arrays,", len(bad), "differ")
    for k in bad[:20]:
        print(k)
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
