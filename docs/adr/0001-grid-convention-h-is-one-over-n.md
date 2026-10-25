# Grid convention: N is the number of cells per side, $h = 1/N$

The finite-difference papers say only "an $N \times N$ grid" and never state the spacing.
On P4 (now T1) at $N = 31$, $h = 1/(N - 1)$ gives $7.617 \times 10^{-5}$, $h = 1/N$ gives $7.144 \times 10^{-5}$ and $h = 1/(N + 1)$ gives $6.696 \times 10^{-5}$, against the published $7.14 \times 10^{-5}$; $h = 1/N$ matches to the printed digits at every published $N$ (`benchmarks/standard_fd.md`, Finding 1).
So every grid method in maesolvers takes `n` as the number of cells per side: $h = 1/n$ on the unit box and $(n + 1)^d$ nodes, and tables label a grid by this $N$.
