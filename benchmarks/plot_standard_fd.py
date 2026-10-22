# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy"]
# ///
"""Draw benchmarks/standard_fd.svg from the CSVs next to this file.

Published values: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3,
column "Standard" (see sources.md). Ours: the h = 1/N rows of
fd_poisson_sequence.csv and fd_newton_sequence.csv.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
PUBLISHED = {
    "P4": {
        31: 7.14e-5,
        45: 3.39e-5,
        63: 1.73e-5,
        89: 0.87e-5,
        127: 0.43e-5,
        181: 0.21e-5,
        255: 0.11e-5,
        361: 0.05e-5,
    },
    "P5": {
        31: 17.15e-3,
        45: 14.59e-3,
        63: 12.53e-3,
        89: 10.67e-3,
        127: 9.00e-3,
        181: 7.59e-3,
        255: 6.42e-3,
        361: 5.41e-3,
    },
}


def read(name, convention_col=None):
    out = {"P4": {}, "P5": {}}
    with open(HERE / name, newline="") as fh:
        for row in csv.DictReader(fh):
            if convention_col and row[convention_col] != "1/N":
                continue
            out[row["problem"]][int(row["N"])] = float(row["max_err_nodes"])
    return out


poisson = read("fd_poisson_sequence.csv")
newton = read("fd_newton_sequence.csv", "h_convention")

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
titles = {
    "P4": "P4: smooth, u = exp(|x|^2/2)",
    "P5": "P5: corner blow-up, u = -sqrt(2 - |x|^2)",
}
for ax, prob in zip(axes, ("P4", "P5")):
    for data, label, style in (
        (
            poisson[prob],
            "FDPoisson (ours)",
            dict(color="#1f77b4", marker="o", ms=4, lw=1.5),
        ),
        (
            newton[prob],
            "FDNewton (ours)",
            dict(color="#d62728", marker="s", ms=4, lw=1.5, ls="--"),
        ),
    ):
        ns = sorted(data)
        ax.loglog(ns, [data[n] for n in ns], label=label, **style)
    ns = sorted(PUBLISHED[prob])
    ax.loglog(
        ns,
        [PUBLISHED[prob][n] for n in ns],
        "kx",
        ms=7,
        mew=1.5,
        label="Froese-Oberman 2011, Table 3",
        zorder=5,
    )
    ax.set_title(titles[prob], fontsize=10)
    ax.set_xlabel("N (cells per side, h = 1/N)")
    ax.set_xticks([31, 63, 127, 255, 361])
    ax.set_xticklabels(["31", "63", "127", "255", "361"])
    ax.minorticks_off()
    ax.grid(True, which="major", lw=0.4, alpha=0.5)
axes[0].set_ylabel("max error at the grid nodes")
axes[1].legend(loc="lower left", frameon=False, fontsize=8)
fig.savefig(HERE / "standard_fd.svg")
print("wrote", HERE / "standard_fd.svg")
