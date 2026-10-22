"""Write benchmarks/t7_bfo2010.csv: T7 against BFO 2010, Table 7.

Benamou-Froese-Oberman, ESAIM: M2AN 44 (2010), Table 7, prints the minimum
of the discrete solution of T7 (f = 1, g = 1 on [-1, 1]^2) at N = 21 to 141,
where N counts nodes per side (standard_fd.md, Finding 3). This script runs
FDPoisson and FDNewton at n = N - 1 cells, where the origin is a node, and at
n = 21, where it is not, logs each run to MLflow (experiments
maesolvers-fd_poisson-T7 and maesolvers-fd_newton-T7) and writes one CSV row
per run.

Run on a100server from the repository root::

    just push && just remote python benchmarks/t7_bfo2010.py
"""

import csv
from pathlib import Path

import mlflow
import numpy as np

from maesolvers import T7, FDNewton, FDPoisson, tracking
from maesolvers.metrics import evaluate

OUT = Path(__file__).with_name("t7_bfo2010.csv")
# Table 7, column "(M1), (M2)": u_min against N, nodes per side.
PUBLISHED = {
    21: 0.2892,
    41: 0.2734,
    61: 0.2682,
    81: 0.2655,
    101: 0.2639,
    121: 0.2629,
    141: 0.2621,
}
# The Poisson iteration needs about n^2 iterations here (BFO 2010, Table 8).
SOLVERS = {
    "fd_poisson": lambda n: FDPoisson(n=n, max_iter=100_000),
    "fd_newton": lambda n: FDNewton(n=n),
}
FIELDS = [
    "solver",
    "N",
    "n",
    "published_u_min",
    "u_origin",
    "u_min_nodes",
    "nonconvex_fraction",
    "eigmin",
    "iterations",
    "converged",
    "time_s",
]


def main() -> None:
    cases = [(N, N - 1) for N in PUBLISHED] + [(21, 21)]
    rows = []
    for name, make in SOLVERS.items():
        for N, n in cases:
            with tracking.run(f"maesolvers-{name}-T7", params={"n": n, "N": N}):
                result = make(n).solve(T7)
                m = evaluate(result)
                row = {
                    "solver": name,
                    "N": N,
                    "n": n,
                    "published_u_min": PUBLISHED[N],
                    "u_origin": m["u_origin"],
                    "u_min_nodes": float(np.min(result.values)),
                    "nonconvex_fraction": m["nonconvex_fraction"],
                    "eigmin": m["eigmin"],
                    "iterations": result.info["iterations"],
                    "converged": result.info["converged"],
                    "time_s": result.info["time"],
                }
                mlflow.log_metrics({k: float(row[k]) for k in FIELDS[3:]})
            rows.append(row)
            print(row, flush=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
